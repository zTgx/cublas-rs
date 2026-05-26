// DGEMM — double-precision matrix-matrix multiply, all variants in one file.
//
// Mirrors `sgemm` with f64. Pascal consumer cards cap FP64 at ~1/32 the FP32
// rate, so don't expect tiled to beat naive by much (sometimes worse — the
// shared-mem traffic is 2× wider per element and the FP64 ALUs are the
// bottleneck). On A100 (datacenter) FP64 is at ~1/2 SP and tiling wins.

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const NAIVE_BLOCK: u32 = 16;
const TILE_SIZE: usize = 16;

// vectorized variant (mirror of sgemm_vectorized)
const VEC_TILE_M: usize = 32;
const VEC_TILE_N: usize = 32;
const VEC_TILE_K: usize = 16;
const VEC_THREAD_M: usize = 4;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn dgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        mut c: DisjointSlice<f64, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < m as usize {
                let n_size = n as usize;
                let k_size = k as usize;
                let mut sum = 0.0f64;
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

    #[kernel]
    pub fn dgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        mut c: DisjointSlice<f64, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f64, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f64, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE_SIZE);
        let mut sum = 0.0f64;
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

    /// Vectorized (thread-coarsened) f64 GEMM. 32×32 block, each thread
    /// computes `VEC_THREAD_M=4` row outputs in one column.
    /// block_dim = (32, 8, 1) = 256 threads. Shared mem: 2 × 32×16 × 8 B = 8 KB.
    #[kernel]
    pub fn dgemm_vectorized(
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        mut c: DisjointSlice<f64>,
    ) {
        static mut TILE_A: SharedArray<f64, 512> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f64, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let block_row = thread::blockIdx_y() as usize * VEC_TILE_M;
        let block_col = thread::blockIdx_x() as usize * VEC_TILE_N;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let col = block_col + tx;
        let row0 = block_row + ty * VEC_THREAD_M;

        let mut acc: [f64; VEC_THREAD_M] = [0.0; VEC_THREAD_M];

        let num_tiles = k_size.div_ceil(VEC_TILE_K);
        let tid = ty * VEC_TILE_N + tx;

        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * VEC_TILE_K;

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

    /// Double-buffered f64 GEMM. Two 16×16 shared-mem buffers ping-pong:
    /// while compute runs on buf k, the next tile loads into buf 1-k.
    /// Shared mem: 2 × 2 × 16×16 × 8 B = 16 KB.
    #[kernel]
    pub fn dgemm_double_buf(
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        mut c: DisjointSlice<f64, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f64, 512> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f64, 512> = SharedArray::UNINIT;

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
        let mut sum = 0.0f64;

        // Prefetch tile 0 into buf 0.
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

        let mut tile = 0usize;
        while tile + 1 < num_tiles {
            let cur_off = (tile & 1) * 256;
            let next_off = ((tile + 1) & 1) * 256;
            let next_start = (tile + 1) * TILE_SIZE;

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

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_naive", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_naive_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &DeviceBuffer<f64>,
    b: &DeviceBuffer<f64>,
    c: &mut DeviceBuffer<f64>,
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
    module.dgemm_naive(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_naive_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_naive(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
    dgemm_naive_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_tiled", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_tiled_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &DeviceBuffer<f64>,
    b: &DeviceBuffer<f64>,
    c: &mut DeviceBuffer<f64>,
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
    module.dgemm_tiled(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_tiled_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_tiled(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
    dgemm_tiled_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- vectorized launchers ----------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_vectorized", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_vectorized_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &DeviceBuffer<f64>,
    b: &DeviceBuffer<f64>,
    c: &mut DeviceBuffer<f64>,
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
    module.dgemm_vectorized(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_vectorized_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_vectorized(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
    dgemm_vectorized_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- double-buffered launchers -----------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_double_buf", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_double_buf_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &DeviceBuffer<f64>,
    b: &DeviceBuffer<f64>,
    c: &mut DeviceBuffer<f64>,
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
    module.dgemm_double_buf(stream, cfg, m as u32, n as u32, k as u32, alpha, a, b, beta, c)?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "dgemm_double_buf_simple", m = config.m, n = config.n, k = config.k),
)]
pub fn dgemm_double_buf(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
    dgemm_double_buf_dev(module, stream, config, &a_dev, &b_dev, &mut c_dev)?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}
