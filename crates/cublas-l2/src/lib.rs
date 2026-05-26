//! BLAS Level 2 — matrix-vector operations.
//!
//! Convention: row-major matrices, host-slice inputs (v1 API).

mod gemv;
mod symv;
pub mod trsv; // public to expose `Triangular` / `Diag` enums

pub use gemv::{dgemv, hgemv, sgemv};
pub use symv::ssymv;
pub use trsv::strsv;
