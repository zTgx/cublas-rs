// DGEMM naive

use cublas_core::GemmConfig;

/// Naive DGEMM kernel launch.
pub fn dgemm_naive(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
    let _ = (config, a, b, c);
    todo!("launch naive DGEMM kernel")
}
