// SGEMM — single-precision matrix-matrix multiply, all variants in one file.
//
// Variants in increasing sophistication:
//   - naive       — one thread per output element, no tiling
//   - tiled       — 16×16 shared-memory tile (cuts global A/B reads ~16×)
//   - vectorized  — 32×32 tile, each thread computes 4 outputs (thread
//                   coarsening: more register accumulators + fewer threads
//                   = better ILP + smaller block-launch overhead)
//   - double_buf  — 16×16 tile, two shared-memory buffers ping-pong: tile
//                   k loads while compute runs on tile k-1 (overlap)

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const NAIVE_BLOCK: u32 = 16;
const TILE_SIZE: usize = 16;

// vectorized variant
const VEC_TILE_M: usize = 32;     // rows per block
const VEC_TILE_N: usize = 32;     // cols per block
const VEC_TILE_K: usize = 16;     // k-direction tile width
const VEC_THREAD_M: usize = 4;    // outputs per thread (rows)
// block_dim = (VEC_TILE_N, VEC_TILE_M / VEC_THREAD_M, 1) = (32, 8, 1) = 256 threads

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Naive: one thread per output element.
    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < m as usize {
                let n_size = n as usize;
                let k_size = k as usize;
                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_size {
                    sum += a[row * k_size + i] * b[i * n_size + col];
                    i += 1;
                }
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }

    /// Tiled: 16×16 shared-memory tile.
    #[kernel]
    pub fn sgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE_SIZE);
        let mut sum = 0.0f32;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;
            let smem_idx = ty * TILE_SIZE + tx;

            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem_idx] = if row < m_size && a_col < k_size {
                    a[row * k_size + a_col]
                } else {
                    0.0
                };
                let b_row = tile_start + ty;
                TILE_B[smem_idx] = if b_row < k_size && col < n_size {
                    b[b_row * n_size + col]
                } else {
                    0.0
                };
            }

            thread::sync_threads();

            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    sum += TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }

    /// Vectorized (thread-coarsened): 32×32 block, each thread computes
    /// `VEC_THREAD_M=4` row outputs in a single column. Inner loop reuses one
    /// B value across 4 register accumulators per k step → ILP gain.
    ///
    /// `block_dim = (32, 8, 1)` (= 256 threads), `grid_dim = (n/32, m/32, 1)`.
    /// Uses 1D `DisjointSlice` indexing (flat) — multiple writes per thread.
    #[kernel]
    pub fn sgemm_vectorized(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32>,
    ) {
        // TILE_A is 32 rows × 16 cols = 512, TILE_B is 16 rows × 32 cols = 512.
        static mut TILE_A: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let block_row = thread::blockIdx_y() as usize * VEC_TILE_M;
        let block_col = thread::blockIdx_x() as usize * VEC_TILE_N;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let col = block_col + tx;
        let row0 = block_row + ty * VEC_THREAD_M;

        // 4 register accumulators — flat array so the compiler can keep them
        // in registers and unroll the inner loop.
        let mut acc: [f32; VEC_THREAD_M] = [0.0; VEC_THREAD_M];

        let num_tiles = k_size.div_ceil(VEC_TILE_K);
        let tid = ty * VEC_TILE_N + tx; // 0..256

        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * VEC_TILE_K;

            // Cooperatively load TILE_A (32 × 16 = 512 elems, 256 threads → 2/thread).
            unsafe {
                let idx0 = tid;
                let r0 = idx0 / VEC_TILE_K;
                let kk0 = idx0 % VEC_TILE_K;
                TILE_A[idx0] = if block_row + r0 < m_size && tile_start + kk0 < k_size {
                    a[(block_row + r0) * k_size + tile_start + kk0]
                } else {
                    0.0
                };
                let idx1 = tid + 256;
                let r1 = idx1 / VEC_TILE_K;
                let kk1 = idx1 % VEC_TILE_K;
                TILE_A[idx1] = if block_row + r1 < m_size && tile_start + kk1 < k_size {
                    a[(block_row + r1) * k_size + tile_start + kk1]
                } else {
                    0.0
                };
            }
            // TILE_B (16 × 32 = 512, 2/thread).
            unsafe {
                let idx0 = tid;
                let kk0 = idx0 / VEC_TILE_N;
                let cc0 = idx0 % VEC_TILE_N;
                TILE_B[idx0] = if tile_start + kk0 < k_size && block_col + cc0 < n_size {
                    b[(tile_start + kk0) * n_size + block_col + cc0]
                } else {
                    0.0
                };
                let idx1 = tid + 256;
                let kk1 = idx1 / VEC_TILE_N;
                let cc1 = idx1 % VEC_TILE_N;
                TILE_B[idx1] = if tile_start + kk1 < k_size && block_col + cc1 < n_size {
                    b[(tile_start + kk1) * n_size + block_col + cc1]
                } else {
                    0.0
                };
            }

            thread::sync_threads();

            // Inner accumulation. For each k, load one B value and multiply
            // into all VEC_THREAD_M=4 accumulators.
            unsafe {
                let mut kk = 0usize;
                while kk < VEC_TILE_K {
                    let b_val = TILE_B[kk * VEC_TILE_N + tx];
                    let mut r = 0usize;
                    while r < VEC_THREAD_M {
                        let a_val = TILE_A[(ty * VEC_THREAD_M + r) * VEC_TILE_K + kk];
                        acc[r] += a_val * b_val;
                        r += 1;
                    }
                    kk += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        // Write VEC_THREAD_M outputs.
        let mut r = 0usize;
        while r < VEC_THREAD_M {
            let g_row = row0 + r;
            if g_row < m_size && col < n_size {
                let c_idx = g_row * n_size + col;
                unsafe {
                    let cur = *c.get_unchecked_mut(c_idx);
                    *c.get_unchecked_mut(c_idx) = alpha * acc[r] + beta * cur;
                }
            }
            r += 1;
        }
    }

    /// Double-buffered: two shared-memory tiles for A and two for B. While
    /// compute runs on buffer k, the next tile is being loaded into buffer
    /// 1−k. One sync per iteration (vs two for `sgemm_tiled`).
    ///
    /// Same block / grid shape as `sgemm_tiled`.
    #[kernel]
    pub fn sgemm_double_buf(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        // Two 16×16 buffers per tile (512 = 2 × 256).
        static mut TILE_A: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE_SIZE);
        if num_tiles == 0 {
            return;
        }

        let smem_idx = ty * TILE_SIZE + tx;
        let mut sum = 0.0f32;

        // Prefetch tile 0 into buf 0 (offset 0).
        unsafe {
            let a_col = tx;
            TILE_A[smem_idx] = if row < m_size && a_col < k_size {
                a[row * k_size + a_col]
            } else {
                0.0
            };
            let b_row = ty;
            TILE_B[smem_idx] = if b_row < k_size && col < n_size {
                b[b_row * n_size + col]
            } else {
                0.0
            };
        }
        thread::sync_threads();

        // Steady-state: for tile 0..num_tiles-1, prefetch (tile+1) and
        // compute on `tile`. One sync per iteration.
        let mut tile = 0usize;
        while tile + 1 < num_tiles {
            let cur_off = (tile & 1) * 256;
            let next_off = ((tile + 1) & 1) * 256;
            let next_start = (tile + 1) * TILE_SIZE;

            // Issue prefetch (no sync after — overlaps with compute below).
            unsafe {
                let a_col = next_start + tx;
                TILE_A[next_off + smem_idx] = if row < m_size && a_col < k_size {
                    a[row * k_size + a_col]
                } else {
                    0.0
                };
                let b_row = next_start + ty;
                TILE_B[next_off + smem_idx] = if b_row < k_size && col < n_size {
                    b[b_row * n_size + col]
                } else {
                    0.0
                };
            }

            // Compute on current tile.
            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    sum += TILE_A[cur_off + ty * TILE_SIZE + i]
                        * TILE_B[cur_off + i * TILE_SIZE + tx];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        // Last tile.
        let last_off = (tile & 1) * 256;
        unsafe {
            let mut i = 0usize;
            while i < TILE_SIZE {
                sum += TILE_A[last_off + ty * TILE_SIZE + i]
                    * TILE_B[last_off + i * TILE_SIZE + tx];
                i += 1;
            }
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }
}

// ---- naive launchers ----------------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_naive", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_naive_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(NAIVE_BLOCK),
            (m as u32).div_ceil(NAIVE_BLOCK),
            1,
        ),
        block_dim: (NAIVE_BLOCK, NAIVE_BLOCK, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch SGEMM naive");
    module.sgemm_naive(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_naive_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_naive(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c)?;
    sgemm_naive_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- tiled launchers ----------------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_tiled", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_tiled_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let tile = TILE_SIZE as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch SGEMM tiled");
    module.sgemm_tiled(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_tiled_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_tiled(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c)?;
    sgemm_tiled_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- vectorized launchers ----------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_vectorized", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_vectorized_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(VEC_TILE_N as u32),
            (m as u32).div_ceil(VEC_TILE_M as u32),
            1,
        ),
        block_dim: (VEC_TILE_N as u32, (VEC_TILE_M / VEC_THREAD_M) as u32, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch SGEMM vectorized");
    module.sgemm_vectorized(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_vectorized_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_vectorized(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c)?;
    sgemm_vectorized_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- double-buffered launchers -----------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_double_buf", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_double_buf_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let tile = TILE_SIZE as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch SGEMM double_buf");
    module.sgemm_double_buf(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "sgemm_double_buf_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn sgemm_double_buf(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c)?;
    sgemm_double_buf_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}
