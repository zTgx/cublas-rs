// HGEMM — half-precision (f16) matrix-matrix multiply.
//
// - `half`         — pure scalar f16 arithmetic (in/out f16, f32 accumulate).
//                    Works on any Pascal+. Stub for now.
// - `tensor_core`  — WMMA / mma.sync. **Blocked**: cuda-oxide does not yet
//                    expose the WMMA intrinsics needed (only WGMMA for Hopper
//                    and tcgen05 for Blackwell are wrapped). Implement when
//                    those intrinsics land, or fork dialect-nvvm locally.

use cublas_core::{GemmConfig, Result};
use half::f16;

/// Scalar HGEMM kernel launch (f16). Stub.
pub fn hgemm_half(config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("launch scalar HGEMM kernel")
}

/// Tensor-Core HGEMM kernel launch (f16, sm_80+). Blocked on WMMA wrapper.
pub fn hgemm_tensor_core(config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("blocked on WMMA wrapper in cuda-oxide")
}
