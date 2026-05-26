//! cuBLAS-rs performance benchmarks.
//!
//! All entry points are currently `todo!()` stubs — they exist so the
//! workspace compiles end-to-end and the shape is locked in.

pub mod batched_gemm;
pub mod gemm;
pub mod hgemm;
pub mod timer;
pub mod validator;
pub mod vector_ops;
