// GEMM API: sgemm / dgemm / hgemm
//
// Unified interface to the GEMM kernel crates.

pub use cublas_dgemm::{dgemm_double_buf, dgemm_naive, dgemm_tiled, dgemm_vectorized};
pub use cublas_hgemm::{half, tensor_core};
pub use cublas_sgemm::{sgemm_double_buf, sgemm_naive, sgemm_tiled, sgemm_vectorized};
