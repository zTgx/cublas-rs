//! SGEMM — single-precision matrix-matrix multiply.
//!
//! Variants in increasing sophistication:
//! - [`naive`] — one thread per output element, no tiling
//! - [`tiled`] — shared-memory tiling
//! - [`vectorized`] — float4 loads + thread coarsening
//! - [`double_buf`] — manual double-buffered shared memory pipeline

mod double_buf;
pub mod naive; // `kernels` submodule consumed by `cublas_l3::Modules::load`
mod tiled;
mod vectorized;

pub use double_buf::sgemm_double_buf;
pub use naive::sgemm_naive;
pub use tiled::sgemm_tiled;
pub use vectorized::sgemm_vectorized;
