// AXPY: y[i] = alpha * x[i] + y[i] (non-f32 variants)
//
// f32 lives in `saxpy.rs` as the reference template implementation.

use cublas_core::Result;
use half::f16;

/// DAXPY — double-precision axpy. Stub.
pub fn daxpy(n: usize, alpha: f64, x: &[f64], y: &mut [f64]) -> Result<()> {
    let _ = (n, alpha, x, y);
    todo!("launch DAXPY kernel")
}

/// HAXPY — half-precision axpy. Stub.
pub fn haxpy(n: usize, alpha: f16, x: &[f16], y: &mut [f16]) -> Result<()> {
    let _ = (n, alpha, x, y);
    todo!("launch HAXPY kernel")
}
