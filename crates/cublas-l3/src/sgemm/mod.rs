//! SGEMM — single-precision matrix-matrix multiply.
//!
//! Variants in increasing sophistication:
//! - [`naive`] — one thread per output element, no tiling
//! - [`tiled`] — shared-memory tiling
//! - [`vectorized`] — float4 loads + thread coarsening (stub)
//! - [`double_buf`] — manual double-buffered shared memory pipeline (stub)

mod double_buf;
pub mod naive; // `kernels` submodule consumed by `cublas_l3::Modules::load`
pub mod tiled; // ditto
mod vectorized;

pub use double_buf::sgemm_double_buf;
pub use naive::{sgemm_naive, sgemm_naive_dev};
pub use tiled::{sgemm_tiled, sgemm_tiled_dev};
pub use vectorized::sgemm_vectorized;
