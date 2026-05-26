// DGEMM naive

use cublas_core::{GemmConfig, Result};

/// Naive DGEMM kernel launch.
pub fn dgemm_naive(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch naive DGEMM kernel")
}
