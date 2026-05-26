// SYMV: y = alpha * A * x + beta * y, A symmetric
//
// Only the specified triangle (Upper or Lower) is read.

use crate::trsv::Triangular;

/// SSYMV — symmetric matrix-vector multiply.
pub fn ssymv(
    uplo: Triangular,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    let _ = (uplo, n, alpha, a, x, beta, y);
    todo!("launch SSYMV kernel")
}
