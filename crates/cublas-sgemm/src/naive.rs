// SGEMM naive: C = alpha * A * B + beta * C
// Direct translation of the mathematical formula, no tiling.

use cublas_core::GemmConfig;

/// Naive SGEMM kernel launch.
///
/// Each thread computes one element of C. No shared memory, no tiling.
/// Useful as a correctness baseline and to measure the cost of uncoalesced access.
///
/// # Arguments
/// * `config` - Matrix dimensions and alpha/beta scalars
/// * `a` - Row-major matrix A of shape (m, k)
/// * `b` - Row-major matrix B of shape (k, n)
/// * `c` - Row-major matrix C of shape (m, n), overwritten with the result
pub fn sgemm_naive(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
    // TODO: implement kernel launch via cuda-oxide
    let _ = (config, a, b, c);
    todo!("launch naive SGEMM kernel")
}
