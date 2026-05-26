// SGEMM vectorized: float4 loads/stores + thread coarsening

use cublas_core::GemmConfig;

/// Vectorized SGEMM kernel launch — wider memory accesses for bandwidth.
pub fn sgemm_vectorized(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
    let _ = (config, a, b, c);
    todo!("launch vectorized SGEMM kernel")
}
