//! # cuBLAS-rs
//!
//! A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
//!
//! ## Usage
//!
//! All types and functions are re-exported at the crate root:
//!
//! ```
//! use cublas_rs::prelude::*;      // common types: GemmConfig, BlasScalar, etc.
//! use cublas_rs::sgemm_naive;     // direct access to kernel functions
//! use cublas_rs::hgemm::hgemm_half; // or via module namespace
//! ```

// Re-export all core types
pub use cublas_core::*;

/// SGEMM module
pub mod sgemm {
    pub use cublas_sgemm::*;
}

/// DGEMM module
pub mod dgemm {
    pub use cublas_dgemm::*;
}

/// HGEMM module
pub mod hgemm {
    pub use cublas_hgemm::*;
}

/// Batched GEMM module
pub mod batched {
    pub use cublas_batched_gemm::*;
}

/// Vector operations module
pub mod vector {
    pub use cublas_vector::*;
}

pub mod prelude {
    //! Common types used by most projects.
    //! `use cublas_rs::prelude::*;` to get started.
    pub use cublas_core::{BlasScalar, GemmConfig, MatrixLayout, Transpose};
}
