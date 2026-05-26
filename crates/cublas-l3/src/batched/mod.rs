//! Batched GEMM — extensions to L3 for running many GEMM ops together.
//!
//! - [`simple::batched_sgemm`] — one CUDA stream per batch. Max concurrency,
//!   high launch overhead. Use when per-batch shapes differ.
//! - [`strided::strided_batched_sgemm`] — single launch over strided storage.
//!   Standard cuBLAS pattern, lowest overhead, identical per-batch shapes.

mod simple;
mod strided;

pub use simple::batched_sgemm;
pub use strided::strided_batched_sgemm;
