//! BLAS Level 3 — matrix-matrix kernels (plus batched extensions).
//!
//! Organised by precision family, each with its own progression of
//! implementation variants (`naive` → `tiled` → `vectorized` → `double_buf`).

pub mod batched;
pub mod dgemm;
pub mod hgemm;
pub mod sgemm;

pub use batched::{batched_sgemm, strided_batched_sgemm};
pub use dgemm::{dgemm_double_buf, dgemm_naive, dgemm_tiled, dgemm_vectorized};
pub use hgemm::{hgemm_half, hgemm_tensor_core};
pub use sgemm::{sgemm_double_buf, sgemm_naive, sgemm_tiled, sgemm_vectorized};
