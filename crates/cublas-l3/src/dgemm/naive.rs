// DGEMM naive: C = alpha * A * B + beta * C   (f64)
//
// Direct copy of the SGEMM naive design but with f64 throughout. On Pascal,
// FP64 runs at ~1/32 the FP32 rate (consumer SKU); on Ampere (A100) it's
// closer to 1/2. Correctness wins for now.

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

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
}

const BLOCK_SIZE: u32 = 16;

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
    let grid_x = (n as u32).div_ceil(BLOCK_SIZE);
    let grid_y = (m as u32).div_ceil(BLOCK_SIZE);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch DGEMM naive");
    module.dgemm_naive(
        stream,
        cfg,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        a,
        b,
        beta,
        c,
    )?;
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
    assert_eq!(a.len(), m * k, "A length must equal m*k");
    assert_eq!(b.len(), k * n, "B length must equal k*n");
    assert_eq!(c.len(), m * n, "C length must equal m*n");
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
