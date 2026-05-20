// DGEMM tiled: shared memory tiling (f64)

use cublas_core::GemmConfig;

/// Tiled DGEMM kernel launch (f64).
pub fn dgemm_tiled(
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let _ = (config, a, b, c);
    todo!("launch tiled DGEMM kernel")
}
