// SAXPY: y[i] = alpha * x[i] + y[i]
//
// Reference implementation — the template every L1 kernel follows.

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn saxpy(alpha: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(y_elem) = y.get_mut(idx) {
            *y_elem = alpha * x[i] + *y_elem;
        }
    }
}

/// Primary path: compute-only on device buffers. Caller owns H2D / D2H.
#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "saxpy"))]
pub fn saxpy_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f32,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch SAXPY");
    module.saxpy(stream, cfg, alpha, x, y)?;
    Ok(())
}

/// Convenience path: takes host slices, allocates, uploads, launches, copies
/// back. Wasteful for hot loops; use `saxpy_dev` with persistent buffers.
#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "saxpy_simple"))]
pub fn saxpy(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f32,
    x: &[f32],
    y: &mut [f32],
) -> Result<()> {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(());
    }
    tracing::trace!("H2D x, y");
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..n])?;
    saxpy_dev(module, stream, n, alpha, &x_dev, &mut y_dev)?;
    tracing::trace!("D2H y");
    let result = y_dev.to_host_vec(stream)?;
    y[..n].copy_from_slice(&result);
    Ok(())
}
