//! # cuBLAS-rs
//!
//! A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
//!
//! ```ignore
//! // Flat — every function pulled up to the crate root
//! use cublas_rs::{saxpy, sgemv, sgemm_naive};
//!
//! // By BLAS level — matches textbook organisation
//! use cublas_rs::level1::saxpy;
//! use cublas_rs::level2::sgemv;
//! use cublas_rs::level3::sgemm_naive;
//!
//! // Common types
//! use cublas_rs::prelude::*;
//! ```

// Shared types (GemmConfig, BlasScalar, MatrixLayout, Transpose)
pub use cublas_core::*;

// Flat re-exports of every BLAS function.
pub use cublas_l1::*;
pub use cublas_l2::*;
pub use cublas_l3::*;

/// BLAS Level 1 — vector operations.
pub mod level1 {
    pub use cublas_l1::*;
}

/// BLAS Level 2 — matrix-vector operations.
pub mod level2 {
    pub use cublas_l2::*;
}

/// BLAS Level 3 — matrix-matrix operations (plus batched extensions).
pub mod level3 {
    pub use cublas_l3::*;
}

pub mod prelude {
    //! Common types used by most callers.
    pub use cublas_core::{BlasScalar, GemmConfig, MatrixLayout, Transpose};
}
