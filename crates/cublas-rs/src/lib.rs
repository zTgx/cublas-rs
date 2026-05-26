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
//!
//! Most L2/L3 methods are still `todo!()` stubs — they panic with a clear
//! message when called. See `CLAUDE.md` for the implementation status table.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DriverError};
use half::f16;

// Shared scalar / matrix types.
pub use cublas_core::*;

// Triangular / Diag enums live in cublas-l2 today; re-export so `strsv` /
// `ssymv` callers don't need to depend on `cublas-l2` directly.
pub use cublas_l2::trsv::{Diag, Triangular};

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
    l3: cublas_l3::Modules,
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
        let l3 = cublas_l3::Modules::load(&ctx)?;
        Ok(Self { ctx, stream, l1, l3 })
    }

    // =====================================================================
    // Level 1 — vector ops
    // =====================================================================

    /// `y := alpha * x + y` (f32). Implemented.
    pub fn saxpy(&self, n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
        cublas_l1::saxpy(&self.l1.saxpy, &self.stream, n, alpha, x, y);
    }

    /// `y := alpha * x + y` (f64).
    pub fn daxpy(&self, n: usize, alpha: f64, x: &[f64], y: &mut [f64]) {
        let _ = (n, alpha, x, y);
        todo!("DAXPY kernel not yet implemented");
    }

    /// `y := alpha * x + y` (f16).
    pub fn haxpy(&self, n: usize, alpha: f16, x: &[f16], y: &mut [f16]) {
        let _ = (n, alpha, x, y);
        todo!("HAXPY kernel not yet implemented");
    }

    /// `x := alpha * x` (in-place).
    pub fn sscal(&self, n: usize, alpha: f32, x: &mut [f32]) {
        let _ = (n, alpha, x);
        todo!("SSCAL kernel not yet implemented");
    }

    /// `y := x` (element-wise copy).
    pub fn scopy(&self, n: usize, x: &[f32], y: &mut [f32]) {
        let _ = (n, x, y);
        todo!("SCOPY kernel not yet implemented");
    }

    /// `sum(x[i] * y[i])`.
    pub fn sdot(&self, n: usize, x: &[f32], y: &[f32]) -> f32 {
        let _ = (n, x, y);
        todo!("SDOT kernel not yet implemented");
    }

    /// `sqrt(sum(x[i]^2))`.
    pub fn snrm2(&self, n: usize, x: &[f32]) -> f32 {
        let _ = (n, x);
        todo!("SNRM2 kernel not yet implemented");
    }

    /// `sum(|x[i]|)`.
    pub fn sasum(&self, n: usize, x: &[f32]) -> f32 {
        let _ = (n, x);
        todo!("SASUM kernel not yet implemented");
    }

    /// Index of `argmax(|x[i]|)`.
    pub fn isamax(&self, n: usize, x: &[f32]) -> usize {
        let _ = (n, x);
        todo!("ISAMAX kernel not yet implemented");
    }

    // =====================================================================
    // Level 2 — matrix-vector ops
    // =====================================================================

    /// `y := alpha * op(A) * x + beta * y` (f32).
    pub fn sgemv(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        let _ = (trans, m, n, alpha, a, x, beta, y);
        todo!("SGEMV kernel not yet implemented");
    }

    /// `y := alpha * op(A) * x + beta * y` (f64).
    pub fn dgemv(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) {
        let _ = (trans, m, n, alpha, a, x, beta, y);
        todo!("DGEMV kernel not yet implemented");
    }

    /// `y := alpha * op(A) * x + beta * y` (f16).
    pub fn hgemv(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f16,
        a: &[f16],
        x: &[f16],
        beta: f16,
        y: &mut [f16],
    ) {
        let _ = (trans, m, n, alpha, a, x, beta, y);
        todo!("HGEMV kernel not yet implemented");
    }

    /// Solve `op(A) * x = b`, A triangular. `b` is overwritten with the solution.
    pub fn strsv(
        &self,
        uplo: Triangular,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[f32],
        b: &mut [f32],
    ) {
        let _ = (uplo, trans, diag, n, a, b);
        todo!("STRSV kernel not yet implemented");
    }

    /// `y := alpha * A * x + beta * y`, A symmetric.
    pub fn ssymv(
        &self,
        uplo: Triangular,
        n: usize,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        let _ = (uplo, n, alpha, a, x, beta, y);
        todo!("SSYMV kernel not yet implemented");
    }

    // =====================================================================
    // Level 3 — matrix-matrix ops
    // =====================================================================

    /// `C := alpha * A * B + beta * C` (f32, naive). Row-major.
    pub fn sgemm_naive(&self, config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        cublas_l3::sgemm_naive(&self.l3.sgemm_naive, &self.stream, config, a, b, c);
    }

    /// SGEMM with shared-memory tiling.
    pub fn sgemm_tiled(&self, config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        let _ = (config, a, b, c);
        todo!("tiled SGEMM kernel not yet implemented");
    }

    /// SGEMM with vectorized (f32x4) loads.
    pub fn sgemm_vectorized(&self, config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        let _ = (config, a, b, c);
        todo!("vectorized SGEMM kernel not yet implemented");
    }

    /// SGEMM with double-buffered shared-memory loads.
    pub fn sgemm_double_buf(&self, config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        let _ = (config, a, b, c);
        todo!("double-buffered SGEMM kernel not yet implemented");
    }

    /// `C := alpha * A * B + beta * C` (f64, naive).
    pub fn dgemm_naive(&self, config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
        let _ = (config, a, b, c);
        todo!("naive DGEMM kernel not yet implemented");
    }

    /// DGEMM with shared-memory tiling.
    pub fn dgemm_tiled(&self, config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
        let _ = (config, a, b, c);
        todo!("tiled DGEMM kernel not yet implemented");
    }

    /// DGEMM with vectorized loads.
    pub fn dgemm_vectorized(&self, config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
        let _ = (config, a, b, c);
        todo!("vectorized DGEMM kernel not yet implemented");
    }

    /// DGEMM with double-buffered shared-memory loads.
    pub fn dgemm_double_buf(&self, config: &GemmConfig<f64>, a: &[f64], b: &[f64], c: &mut [f64]) {
        let _ = (config, a, b, c);
        todo!("double-buffered DGEMM kernel not yet implemented");
    }

    /// `C := alpha * A * B + beta * C` (f16 scalar path).
    pub fn hgemm_half(&self, config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) {
        let _ = (config, a, b, c);
        todo!("scalar HGEMM kernel not yet implemented");
    }

    /// HGEMM via Tensor Cores. Blocked on a WMMA wrapper in cuda-oxide.
    pub fn hgemm_tensor_core(&self, config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) {
        let _ = (config, a, b, c);
        todo!("Tensor-Core HGEMM blocked on cuda-oxide WMMA support");
    }

    /// Batched SGEMM — one stream per batch.
    pub fn batched_sgemm(
        &self,
        config: &GemmConfig<f32>,
        batch_count: usize,
        a: &[&[f32]],
        b: &[&[f32]],
        c: &mut [&mut [f32]],
    ) {
        let _ = (config, batch_count, a, b, c);
        todo!("batched SGEMM not yet implemented");
    }

    /// Strided batched SGEMM — single launch over strided storage.
    pub fn strided_batched_sgemm(
        &self,
        config: &GemmConfig<f32>,
        batch_count: usize,
        a: &[f32],
        stride_a: usize,
        b: &[f32],
        stride_b: usize,
        c: &mut [f32],
        stride_c: usize,
    ) {
        let _ = (config, batch_count, a, stride_a, b, stride_b, c, stride_c);
        todo!("strided batched SGEMM not yet implemented");
    }
}

pub mod prelude {
    //! Common types used by most callers.
    pub use crate::{Diag, Handle, Triangular};
    pub use cublas_core::{BlasScalar, GemmConfig, MatrixLayout, Transpose};
}
