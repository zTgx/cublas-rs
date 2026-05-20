// DGEMM vectorized (f64)

use cublas_core::GemmConfig;

/// Vectorized DGEMM kernel launch (f64).
pub fn dgemm_vectorized(
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let _ = (config, a, b, c);
    todo!("launch vectorized DGEMM kernel")
}
