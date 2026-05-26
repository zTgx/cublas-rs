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

// ---- stubs --------------------------------------------------------------

/// DGEMM with vectorized loads. Stub.
pub fn dgemm_vectorized(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch vectorized DGEMM kernel")
}

/// DGEMM with double-buffered shared-memory loads. Stub.
pub fn dgemm_double_buf(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch double-buffered DGEMM kernel")
}
