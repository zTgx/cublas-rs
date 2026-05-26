// TRSV: solve op(A) * x = b in place (x overwrites b)
//
// A is n x n triangular (row-major). Gates Cholesky / LU solvers.

use cublas_core::{Result, Transpose};

/// Whether A is upper- or lower-triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Triangular {
    Upper,
    Lower,
}

/// Whether the triangular matrix has a unit (implicit 1) diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    NonUnit,
    Unit,
}

/// STRSV — solve op(A) * x = b where A is triangular.
///
/// `b` is overwritten with the solution `x`.
pub fn strsv(
    uplo: Triangular,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f32],
    b: &mut [f32],
) -> Result<()> {
    let _ = (uplo, trans, diag, n, a, b);
    todo!("launch STRSV kernel")
}
