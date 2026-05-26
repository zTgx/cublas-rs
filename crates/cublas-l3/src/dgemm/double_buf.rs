// DGEMM double buffer

use cublas_core::{GemmConfig, Result};

/// Double-buffered DGEMM kernel launch.
pub fn dgemm_double_buf(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch double-buffered DGEMM kernel")
}
