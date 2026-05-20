// HGEMM scalar: f16 pure scalar implementation
//
// All arithmetic in f16. No Tensor Core usage.

use cublas_core::GemmConfig;
use half::f16;

/// Scalar HGEMM kernel launch (f16).
pub fn hgemm_half(
    config: &GemmConfig<f16>,
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
) {
    let _ = (config, a, b, c);
    todo!("launch scalar HGEMM kernel")
}
