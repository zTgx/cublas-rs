// SGEMM tiled: shared-memory tiling (16x16 / 32x32)

use cublas_core::GemmConfig;

/// Tiled SGEMM kernel launch — tiles A and B into shared memory.
pub fn sgemm_tiled(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
    let _ = (config, a, b, c);
    todo!("launch tiled SGEMM kernel")
}
