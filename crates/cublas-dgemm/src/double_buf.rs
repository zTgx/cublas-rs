// DGEMM double buffer (f64)

use cublas_core::GemmConfig;

/// Double-buffered DGEMM kernel launch (f64).
pub fn dgemm_double_buf(
    config: &GemmConfig<f64>,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let _ = (config, a, b, c);
    todo!("launch double-buffered DGEMM kernel")
}
