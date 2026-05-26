// Strided batched SGEMM: contiguous buffer per matrix-set
//
// Each matrix A_k = a[k * stride_a ..]. Same for B and C.

use cublas_core::{GemmConfig, Result};

/// Strided batched SGEMM kernel launch.
///
/// All A_k share `(m, k)`, all B_k share `(k, n)`, all C_k share `(m, n)`.
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
