// SGEMM — single-precision matrix-matrix multiply, all variants in one file.
//
// Variants in increasing sophistication:
//   - naive       — one thread per output element, no tiling
//   - tiled       — 16×16 shared-memory tile (cuts global A/B reads ~16×)
//   - vectorized  — float4 loads + thread coarsening    (stub)
//   - double_buf  — manual double-buffered shared mem   (stub)

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const NAIVE_BLOCK: u32 = 16;
const TILE_SIZE: usize = 16;

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

// ---- stubs --------------------------------------------------------------

/// SGEMM with vectorized (f32x4) loads. Stub.
pub fn sgemm_vectorized(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch vectorized SGEMM kernel")
}

/// SGEMM with double-buffered shared-memory loads. Stub.
pub fn sgemm_double_buf(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch double-buffered SGEMM kernel")
}
