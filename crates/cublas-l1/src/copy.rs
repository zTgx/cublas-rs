// SCOPY: y[i] = x[i]

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn scopy(x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(y_elem) = y.get_mut(idx) {
            *y_elem = x[i];
        }
    }
}

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "scopy"))]
pub fn scopy_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);
    module.scopy(stream, cfg, x, y)?;
    Ok(())
}

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "scopy_simple"))]
pub fn scopy(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<()> {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(());
    }
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..n])?;
    scopy_dev(module, stream, n, &x_dev, &mut y_dev)?;
    let result = y_dev.to_host_vec(stream)?;
    y[..n].copy_from_slice(&result);
    Ok(())
}
