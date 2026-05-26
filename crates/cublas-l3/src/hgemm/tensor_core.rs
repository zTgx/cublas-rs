// HGEMM Tensor Core: WMMA / mma.sync path
//
// BLOCKED: cuda-oxide does not yet expose the wmma::* / mma.sync intrinsics
// (only WGMMA for Hopper and tcgen05 for Blackwell are wrapped). Implement
// when those intrinsics land, or fork dialect-nvvm locally.

use cublas_core::GemmConfig;
use half::f16;

/// Tensor Core HGEMM kernel launch (f16, sm_80+).
pub fn hgemm_tensor_core(config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) {
    let _ = (config, a, b, c);
    todo!("blocked on WMMA wrapper in cuda-oxide")
}
