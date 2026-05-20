// HGEMM Tensor Core: WMMA/MMA instruction implementation
//
// Uses Tensor Core via WMMA (Warp Matrix Multiply-Accumulate) or
// MMA (Matrix Multiply-Accumulate) PTX instructions.

use cublas_core::GemmConfig;
use half::f16;

/// Tensor Core HGEMM kernel launch (f16).
pub fn hgemm_tensor_core(config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) {
    let _ = (config, a, b, c);
    todo!("launch Tensor Core HGEMM kernel")
}
