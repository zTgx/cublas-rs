//! DGEMM — double-precision matrix-matrix multiply.
//!
//! Same four variants as SGEMM. Note: Ampere has FP64 Tensor Cores (A100) but
//! the WMMA wrapper that would expose them is missing in cuda-oxide today.

mod double_buf;
pub mod naive; // `kernels` submodule consumed by `cublas_l3::Modules::load`
pub mod tiled; // ditto
mod vectorized;

pub use double_buf::dgemm_double_buf;
pub use naive::{dgemm_naive, dgemm_naive_dev};
pub use tiled::{dgemm_tiled, dgemm_tiled_dev};
pub use vectorized::dgemm_vectorized;
