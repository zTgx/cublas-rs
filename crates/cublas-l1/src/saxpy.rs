// SAXPY: y[i] = alpha * x[i] + y[i]
//
// Reference implementation — the template every L1 kernel follows.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
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

/// Computes `y[i] = alpha * x[i] + y[i]` for `i in 0..n`.
///
/// # Panics
/// If `x.len()` or `y.len()` is less than `n`, or CUDA initialization fails.
pub fn saxpy(n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return;
    }

    let ctx = CudaContext::new(0).expect("CUDA context init");
    let stream = ctx.default_stream();

    let x_dev = DeviceBuffer::from_host(&stream, &x[..n]).expect("copy x to device");
    let mut y_dev = DeviceBuffer::from_host(&stream, &y[..n]).expect("copy y to device");

    let module = kernels::load(&ctx).expect("load SAXPY PTX");
    module
        .saxpy(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            alpha,
            &x_dev,
            &mut y_dev,
        )
        .expect("SAXPY launch");

    let result = y_dev.to_host_vec(&stream).expect("copy y back");
    y[..n].copy_from_slice(&result);
}
