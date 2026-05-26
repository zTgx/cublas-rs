// DGEMM tiled

use cublas_core::{GemmConfig, Result};

/// Tiled DGEMM kernel launch.
pub fn dgemm_tiled(config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch tiled DGEMM kernel")
}
