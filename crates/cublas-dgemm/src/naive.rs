// DGEMM naive: C = alpha * A * B + beta * C (f64)
// Direct translation of the mathematical formula, no tiling.

use cublas_core::GemmConfig;

/// Naive DGEMM kernel launch (f64).
pub fn dgemm_naive(
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let _ = (config, a, b, c);
    todo!("launch naive DGEMM kernel")
}
