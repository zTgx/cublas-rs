//! # cuBLAS-rs
//!
//! A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
//!
//! Modelled after the C cuBLAS API: build a [`Handle`] once, then call
//! BLAS ops as methods on it.
//!
//! ```ignore
//! let h = cublas_rs::Handle::new()?;
//! h.saxpy(n, alpha, &x, &mut y);
//! ```

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DriverError};

// Shared types (GemmConfig, BlasScalar, MatrixLayout, Transpose).
pub use cublas_core::*;

// L2/L3 ops are still free-function stubs (each `todo!()`). They will move
// onto `Handle` as their kernels get implemented.
pub use cublas_l2::*;
pub use cublas_l3::*;

/// Owns the CUDA context, default stream, and loaded kernel modules. Mirrors
/// `cublasHandle_t` from the C cuBLAS API — build once, reuse across calls.
///
/// PTX files (`cublas_l1.ptx`, ...) are loaded from the current working
/// directory; cargo-oxide drops them at the workspace root. Run binaries
/// that use `Handle` from the workspace root.
pub struct Handle {
    #[allow(dead_code)] // held to keep CUDA context alive for stream + modules
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    l1: cublas_l1::Modules,
}

impl Handle {
    /// Initialise on CUDA device 0.
    pub fn new() -> Result<Self, DriverError> {
        Self::with_device(0)
    }

    /// Initialise on a specific CUDA device.
    pub fn with_device(device_idx: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(device_idx)?;
        let stream = ctx.default_stream();
        let l1 = cublas_l1::Modules::load(&ctx)?;
        Ok(Self { ctx, stream, l1 })
    }

    // ---- Level 1 ----

    /// `y := alpha * x + y`
    pub fn saxpy(&self, n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
        cublas_l1::saxpy(&self.l1.saxpy, &self.stream, n, alpha, x, y);
    }
}

pub mod prelude {
    //! Common types used by most callers.
    pub use crate::Handle;
    pub use cublas_core::{BlasScalar, GemmConfig, MatrixLayout, Transpose};
}
