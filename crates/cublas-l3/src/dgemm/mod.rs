//! DGEMM — double-precision matrix-matrix multiply.
//!
//! Same four variants as SGEMM. Note: Ampere has FP64 Tensor Cores (A100) but
//! the WMMA wrapper that would expose them is missing in cuda-oxide today.

mod double_buf;
mod naive;
mod tiled;
mod vectorized;

pub use double_buf::dgemm_double_buf;
pub use naive::dgemm_naive;
pub use tiled::dgemm_tiled;
pub use vectorized::dgemm_vectorized;
