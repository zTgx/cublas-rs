// DGEMM vectorized

use cublas_core::GemmConfig;

/// Vectorized DGEMM kernel launch.
pub fn dgemm_vectorized(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
    let _ = (config, a, b, c);
    todo!("launch vectorized DGEMM kernel")
}
