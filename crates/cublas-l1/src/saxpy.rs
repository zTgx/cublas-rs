// SAXPY: y[i] = alpha * x[i] + y[i]
//
// Reference implementation — the template every L1 kernel follows.
//
// The kernel `mod kernels` is `pub` so a binary in another crate can call
// `kernels::from_module` on a `CudaModule` loaded from `cublas_l1.ptx`.
// `kernels::load(ctx)` only works from a binary in *this* crate, because the
// embedded artifact bundle is only present in the entry crate's `.oxart`
// section. See CLAUDE.md → "Calling L1/L2/L3 ops from another crate".

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

/// Computes `y[i] = alpha * x[i] + y[i]` for `i in 0..n`.
///
/// The caller owns the CUDA context, stream, and typed kernel module — load
/// the PTX once at program start and reuse across calls. See
/// `examples/saxpy.rs` for the call pattern.
///
/// # Panics
/// If `x.len()` or `y.len()` is less than `n`, or any CUDA call fails.
pub fn saxpy(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f32,
    x: &[f32],
    y: &mut [f32],
) {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return;
    }

    let x_dev = DeviceBuffer::from_host(stream, &x[..n]).expect("copy x to device");
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..n]).expect("copy y to device");

    module
        .saxpy(
            stream,
            LaunchConfig::for_num_elems(n as u32),
            alpha,
            &x_dev,
            &mut y_dev,
        )
        .expect("SAXPY launch");

    let result = y_dev.to_host_vec(stream).expect("copy y back");
    y[..n].copy_from_slice(&result);
}
