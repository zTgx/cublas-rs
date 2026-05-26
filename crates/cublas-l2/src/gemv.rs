// GEMV: y = alpha * A * x + beta * y
//
// A is m x n (row-major). x is length n. y is length m.

use cublas_core::Transpose;
use half::f16;

/// SGEMV — single-precision matrix-vector multiply.
///
/// Computes y := alpha * op(A) * x + beta * y, where op(A) is A or A^T.
pub fn sgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    let _ = (trans, m, n, alpha, a, x, beta, y);
    todo!("launch SGEMV kernel")
}

/// DGEMV — double-precision matrix-vector multiply.
pub fn dgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    let _ = (trans, m, n, alpha, a, x, beta, y);
    todo!("launch DGEMV kernel")
}

/// HGEMV — half-precision matrix-vector multiply.
pub fn hgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f16,
    a: &[f16],
    x: &[f16],
    beta: f16,
    y: &mut [f16],
) {
    let _ = (trans, m, n, alpha, a, x, beta, y);
    todo!("launch HGEMV kernel")
}
