// SGEMM naive: C = alpha * A * B + beta * C
// One thread per element of C. No shared memory.

use cublas_core::GemmConfig;

/// Naive SGEMM kernel launch — baseline for correctness and bandwidth.
pub fn sgemm_naive(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
    let _ = (config, a, b, c);
    todo!("launch naive SGEMM kernel")
}
