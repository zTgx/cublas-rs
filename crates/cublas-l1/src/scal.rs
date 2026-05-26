// SSCAL: x[i] = alpha * x[i]

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sscal(alpha: f32, mut x: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(x_elem) = x.get_mut(idx) {
            *x_elem = alpha * (*x_elem);
        }
    }
}

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "sscal"))]
pub fn sscal_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f32,
    x: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);
    module.sscal(stream, cfg, alpha, x)?;
    Ok(())
}

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "sscal_simple"))]
pub fn sscal(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f32,
    x: &mut [f32],
) -> Result<()> {
    assert!(x.len() >= n, "x is shorter than n");
    if n == 0 {
        return Ok(());
    }
    let mut x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    sscal_dev(module, stream, n, alpha, &mut x_dev)?;
    let result = x_dev.to_host_vec(stream)?;
    x[..n].copy_from_slice(&result);
    Ok(())
}
