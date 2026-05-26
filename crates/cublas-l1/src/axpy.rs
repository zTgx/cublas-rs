// AXPY: y[i] = alpha * x[i] + y[i] (non-f32 variants)
//
// f32 lives in `saxpy.rs` as the reference template implementation.

use half::f16;

/// DAXPY — double-precision axpy.
pub fn daxpy(n: usize, alpha: f64, x: &[f64], y: &mut [f64]) {
    let _ = (n, alpha, x, y);
    todo!("launch DAXPY kernel")
}

/// HAXPY — half-precision axpy.
pub fn haxpy(n: usize, alpha: f16, x: &[f16], y: &mut [f16]) {
    let _ = (n, alpha, x, y);
    todo!("launch HAXPY kernel")
}
