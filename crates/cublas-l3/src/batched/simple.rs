// Batched GEMM: one CUDA stream per batch

use cublas_core::GemmConfig;

/// Batched SGEMM kernel launch.
///
/// Executes `batch_count` independent GEMM operations concurrently — each
/// batch on its own CUDA stream.
pub fn batched_sgemm(
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &[&[f32]],
    b: &[&[f32]],
    c: &mut [&mut [f32]],
) {
    let _ = (config, batch_count, a, b, c);
    todo!("launch batched SGEMM kernel")
}
