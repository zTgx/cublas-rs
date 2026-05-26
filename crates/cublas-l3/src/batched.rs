// Batched GEMM — extensions to L3 for running many GEMM ops together.
//
// - `batched_sgemm`           — one CUDA stream per batch. Max concurrency,
//                               high launch overhead. Use when per-batch
//                               shapes differ.
// - `strided_batched_sgemm`   — single launch over strided storage. Standard
//                               cuBLAS pattern, lowest overhead, identical
//                               per-batch shapes.

use cublas_core::{GemmConfig, Result};

/// Batched SGEMM kernel launch. Stub.
pub fn batched_sgemm(
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &[&[f32]],
    b: &[&[f32]],
    c: &mut [&mut [f32]],
) -> Result<()> {
    let _ = (config, batch_count, a, b, c);
    todo!("launch batched SGEMM kernel")
}

/// Strided batched SGEMM kernel launch. Stub.
///
/// Each matrix `A_k = a[k * stride_a ..]`. Same for B and C.
pub fn strided_batched_sgemm(
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &[f32],
    stride_a: usize,
    b: &[f32],
    stride_b: usize,
    c: &mut [f32],
    stride_c: usize,
) -> Result<()> {
    let _ = (config, batch_count, a, stride_a, b, stride_b, c, stride_c);
    todo!("launch strided batched SGEMM kernel")
}
