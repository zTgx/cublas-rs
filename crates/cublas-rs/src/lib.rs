//! # cuBLAS-rs
//!
//! A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
//!
//! Modelled after the C cuBLAS API: build a [`Handle`] once, then call BLAS
//! ops as methods on it. Every op returns [`Result`]; the error type is
//! [`CublasError`].
//!
//! ```ignore
//! let h = cublas_rs::Handle::new()?;
//! h.saxpy(n, alpha, &x, &mut y)?;
//! ```
//!
//! Unimplemented L2 / DGEMM / HGEMM / batched methods keep their signature
//! but panic via `todo!()` when called. See `CLAUDE.md` for the
//! implementation status table.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, device_buffer::DeviceCopy};
use half::f16;

// Shared scalar / matrix types + the unified error type.
pub use cublas_core::{
    BlasScalar, CublasError, GemmConfig, MatrixLayout, Result, Transpose,
};

// Triangular / Diag enums live in cublas-l2 today; re-export so `strsv` /
// `ssymv` callers don't need to depend on `cublas-l2` directly.
pub use cublas_l2::trsv::{Diag, Triangular};

// Re-export `DeviceBuffer` so the device-buffer API surface lives entirely
// on `cublas_rs::*`.
pub use cuda_core::DeviceBuffer as DeviceBuf;

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
    l2: cublas_l2::Modules,
    l3: cublas_l3::Modules,
}

impl Handle {
    /// Initialise on CUDA device 0.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialise on a specific CUDA device.
    #[tracing::instrument(level = "info", name = "Handle::with_device")]
    pub fn with_device(device_idx: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_idx)?;
        let stream = ctx.default_stream();
        tracing::info!(device_idx, "CUDA context + default stream ready");
        let l1 = cublas_l1::Modules::load(&ctx)?;
        let l2 = cublas_l2::Modules::load(&ctx)?;
        let l3 = cublas_l3::Modules::load(&ctx)?;
        tracing::info!("Handle ready (L1 + L2 + L3 modules loaded)");
        Ok(Self { ctx, stream, l1, l2, l3 })
    }

    /// Direct access to the underlying CUDA stream. Useful when interleaving
    /// cublas-rs ops with custom kernels.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Block the calling thread until all enqueued work on this handle's
    /// stream completes. Essential when timing — kernel launches are async,
    /// so without a sync your timer just measures the launch overhead.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    // =====================================================================
    // Memory helpers
    // =====================================================================

    /// Copy a host slice onto the GPU. Returned buffer is bound to this
    /// handle's stream.
    pub fn upload<T: DeviceCopy>(&self, host: &[T]) -> Result<DeviceBuffer<T>> {
        Ok(DeviceBuffer::from_host(&self.stream, host)?)
    }

    /// Copy a device buffer back to a fresh `Vec`.
    pub fn download<T: DeviceCopy>(&self, dev: &DeviceBuffer<T>) -> Result<Vec<T>> {
        Ok(dev.to_host_vec(&self.stream)?)
    }

    // =====================================================================
    // Level 1 — vector ops
    //
    // Each op exists in two forms:
    //   - `*_simple` takes host slices (allocates + H2D + launch + D2H).
    //     Convenience path; one-shot use.
    //   - The unsuffixed method takes `&DeviceBuffer<T>` (compute only).
    //     Production path; amortises transfers across many calls.
    // =====================================================================

    /// `y := alpha * x + y` (f32, device buffers).
    pub fn saxpy(
        &self,
        n: usize,
        alpha: f32,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l1::saxpy_dev(&self.l1.saxpy, &self.stream, n, alpha, x, y)
    }

    /// `y := alpha * x + y` (f32, host slices — uploads + downloads).
    pub fn saxpy_simple(
        &self,
        n: usize,
        alpha: f32,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        cublas_l1::saxpy(&self.l1.saxpy, &self.stream, n, alpha, x, y)
    }

    /// `x := alpha * x` (f32, in-place, device buffer).
    pub fn sscal(&self, n: usize, alpha: f32, x: &mut DeviceBuffer<f32>) -> Result<()> {
        cublas_l1::sscal_dev(&self.l1.sscal, &self.stream, n, alpha, x)
    }

    /// `x := alpha * x` (f32, in-place, host slice).
    pub fn sscal_simple(&self, n: usize, alpha: f32, x: &mut [f32]) -> Result<()> {
        cublas_l1::sscal(&self.l1.sscal, &self.stream, n, alpha, x)
    }

    /// `y := x` (f32 copy, device buffers).
    pub fn scopy(
        &self,
        n: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l1::scopy_dev(&self.l1.scopy, &self.stream, n, x, y)
    }

    /// `y := x` (f32 copy, host slices).
    pub fn scopy_simple(&self, n: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        cublas_l1::scopy(&self.l1.scopy, &self.stream, n, x, y)
    }

    /// `sum(x[i] * y[i])` (f32 dot, device buffers).
    pub fn sdot(&self, n: usize, x: &DeviceBuffer<f32>, y: &DeviceBuffer<f32>) -> Result<f32> {
        cublas_l1::sdot_dev(&self.l1.sdot, &self.stream, n, x, y)
    }

    /// `sum(x[i] * y[i])` (f32 dot, host slices).
    pub fn sdot_simple(&self, n: usize, x: &[f32], y: &[f32]) -> Result<f32> {
        cublas_l1::sdot(&self.l1.sdot, &self.stream, n, x, y)
    }

    /// `sqrt(sum(x[i]^2))` (f32, device buffer).
    pub fn snrm2(&self, n: usize, x: &DeviceBuffer<f32>) -> Result<f32> {
        cublas_l1::snrm2_dev(&self.l1.snrm2, &self.stream, n, x)
    }

    /// `sqrt(sum(x[i]^2))` (f32, host slice).
    pub fn snrm2_simple(&self, n: usize, x: &[f32]) -> Result<f32> {
        cublas_l1::snrm2(&self.l1.snrm2, &self.stream, n, x)
    }

    /// `sum(|x[i]|)` (f32, device buffer).
    pub fn sasum(&self, n: usize, x: &DeviceBuffer<f32>) -> Result<f32> {
        cublas_l1::sasum_dev(&self.l1.sasum, &self.stream, n, x)
    }

    /// `sum(|x[i]|)` (f32, host slice).
    pub fn sasum_simple(&self, n: usize, x: &[f32]) -> Result<f32> {
        cublas_l1::sasum(&self.l1.sasum, &self.stream, n, x)
    }

    /// `argmax(|x[i]|)` (f32, device buffer).
    pub fn isamax(&self, n: usize, x: &DeviceBuffer<f32>) -> Result<usize> {
        cublas_l1::isamax_dev(&self.l1.isamax, &self.stream, n, x)
    }

    /// `argmax(|x[i]|)` (f32, host slice).
    pub fn isamax_simple(&self, n: usize, x: &[f32]) -> Result<usize> {
        cublas_l1::isamax(&self.l1.isamax, &self.stream, n, x)
    }

    /// `y := alpha * x + y` (f64). Stub.
    pub fn daxpy(&self, n: usize, alpha: f64, x: &[f64], y: &mut [f64]) -> Result<()> {
        let _ = (n, alpha, x, y);
        todo!("DAXPY kernel not yet implemented");
    }

    /// `y := alpha * x + y` (f16). Stub.
    pub fn haxpy(&self, n: usize, alpha: f16, x: &[f16], y: &mut [f16]) -> Result<()> {
        let _ = (n, alpha, x, y);
        todo!("HAXPY kernel not yet implemented");
    }

    // =====================================================================
    // Level 2 — matrix-vector ops (all stubs)
    // =====================================================================

    /// `y := alpha * op(A) * x + beta * y` (f32, device buffers).
    /// `op(A)` is `A` (NoTrans) or `Aᵀ` (Trans). A is m×n row-major.
    pub fn sgemv(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        beta: f32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l2::sgemv_dev(&self.l2.gemv, &self.stream, trans, m, n, alpha, a, x, beta, y)
    }

    /// `y := alpha * op(A) * x + beta * y` (f32, host slices).
    pub fn sgemv_simple(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> Result<()> {
        cublas_l2::sgemv(&self.l2.gemv, &self.stream, trans, m, n, alpha, a, x, beta, y)
    }

    /// `y := alpha * op(A) * x + beta * y` (f32, shared-mem tiled, device buffers).
    pub fn sgemv_tiled(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        beta: f32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l2::sgemv_tiled_dev(
            &self.l2.gemv,
            &self.stream,
            trans,
            m,
            n,
            alpha,
            a,
            x,
            beta,
            y,
        )
    }

    /// `y := alpha * op(A) * x + beta * y` (f32, shared-mem tiled, host slices).
    pub fn sgemv_tiled_simple(
        &self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> Result<()> {
        cublas_l2::sgemv_tiled(
            &self.l2.gemv,
            &self.stream,
            trans,
            m,
            n,
            alpha,
            a,
            x,
            beta,
            y,
        )
    }

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
    ) -> Result<()> {
        let _ = (trans, m, n, alpha, a, x, beta, y);
        todo!("DGEMV kernel not yet implemented");
    }

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
    ) -> Result<()> {
        let _ = (trans, m, n, alpha, a, x, beta, y);
        todo!("HGEMV kernel not yet implemented");
    }

    /// Solve `op(A) * x = b` in place (`b` overwritten with solution).
    /// Single-thread sequential kernel — correct on any arch, slow for big n.
    pub fn strsv(
        &self,
        uplo: Triangular,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &DeviceBuffer<f32>,
        b: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l2::strsv_dev(&self.l2.trsv, &self.stream, uplo, trans, diag, n, a, b)
    }

    /// Solve `op(A) * x = b` (host slices).
    pub fn strsv_simple(
        &self,
        uplo: Triangular,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &[f32],
        b: &mut [f32],
    ) -> Result<()> {
        cublas_l2::strsv(&self.l2.trsv, &self.stream, uplo, trans, diag, n, a, b)
    }

    /// `y := alpha * A * x + beta * y`, A symmetric (only `uplo` half read).
    pub fn ssymv(
        &self,
        uplo: Triangular,
        n: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        beta: f32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l2::ssymv_dev(
            &self.l2.symv,
            &self.stream,
            uplo,
            n,
            alpha,
            a,
            x,
            beta,
            y,
        )
    }

    /// `y := alpha * A * x + beta * y`, A symmetric (host slices).
    pub fn ssymv_simple(
        &self,
        uplo: Triangular,
        n: usize,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> Result<()> {
        cublas_l2::ssymv(&self.l2.symv, &self.stream, uplo, n, alpha, a, x, beta, y)
    }

    // =====================================================================
    // Level 3 — matrix-matrix ops
    // =====================================================================

    /// `C := alpha * A * B + beta * C` (f32, naive, device buffers).
    pub fn sgemm_naive(
        &self,
        config: &GemmConfig<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l3::sgemm_naive_dev(&self.l3.sgemm_naive, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f32, naive, host slices).
    pub fn sgemm_naive_simple(
        &self,
        config: &GemmConfig<f32>,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        cublas_l3::sgemm_naive(&self.l3.sgemm_naive, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f32, shared-memory tiled, device buffers).
    pub fn sgemm_tiled(
        &self,
        config: &GemmConfig<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        cublas_l3::sgemm_tiled_dev(&self.l3.sgemm_tiled, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f32, shared-memory tiled, host slices).
    pub fn sgemm_tiled_simple(
        &self,
        config: &GemmConfig<f32>,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        cublas_l3::sgemm_tiled(&self.l3.sgemm_tiled, &self.stream, config, a, b, c)
    }

    pub fn sgemm_vectorized(
        &self,
        config: &GemmConfig<f32>,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("vectorized SGEMM kernel not yet implemented");
    }

    pub fn sgemm_double_buf(
        &self,
        config: &GemmConfig<f32>,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("double-buffered SGEMM kernel not yet implemented");
    }

    /// `C := alpha * A * B + beta * C` (f64, naive, device buffers).
    pub fn dgemm_naive(
        &self,
        config: &GemmConfig<f64>,
        a: &DeviceBuffer<f64>,
        b: &DeviceBuffer<f64>,
        c: &mut DeviceBuffer<f64>,
    ) -> Result<()> {
        cublas_l3::dgemm_naive_dev(&self.l3.dgemm_naive, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f64, naive, host slices).
    pub fn dgemm_naive_simple(
        &self,
        config: &GemmConfig<f64>,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
    ) -> Result<()> {
        cublas_l3::dgemm_naive(&self.l3.dgemm_naive, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f64, shared-mem tiled, device buffers).
    pub fn dgemm_tiled(
        &self,
        config: &GemmConfig<f64>,
        a: &DeviceBuffer<f64>,
        b: &DeviceBuffer<f64>,
        c: &mut DeviceBuffer<f64>,
    ) -> Result<()> {
        cublas_l3::dgemm_tiled_dev(&self.l3.dgemm_tiled, &self.stream, config, a, b, c)
    }

    /// `C := alpha * A * B + beta * C` (f64, shared-mem tiled, host slices).
    pub fn dgemm_tiled_simple(
        &self,
        config: &GemmConfig<f64>,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
    ) -> Result<()> {
        cublas_l3::dgemm_tiled(&self.l3.dgemm_tiled, &self.stream, config, a, b, c)
    }

    pub fn dgemm_vectorized(
        &self,
        config: &GemmConfig<f64>,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("vectorized DGEMM kernel not yet implemented");
    }

    pub fn dgemm_double_buf(
        &self,
        config: &GemmConfig<f64>,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("double-buffered DGEMM kernel not yet implemented");
    }

    pub fn hgemm_half(
        &self,
        config: &GemmConfig<f16>,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("scalar HGEMM kernel not yet implemented");
    }

    pub fn hgemm_tensor_core(
        &self,
        config: &GemmConfig<f16>,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        let _ = (config, a, b, c);
        todo!("Tensor-Core HGEMM blocked on cuda-oxide WMMA support");
    }

    pub fn batched_sgemm(
        &self,
        config: &GemmConfig<f32>,
        batch_count: usize,
        a: &[&[f32]],
        b: &[&[f32]],
        c: &mut [&mut [f32]],
    ) -> Result<()> {
        let _ = (config, batch_count, a, b, c);
        todo!("batched SGEMM not yet implemented");
    }

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
    ) -> Result<()> {
        let _ = (config, batch_count, a, stride_a, b, stride_b, c, stride_c);
        todo!("strided batched SGEMM not yet implemented");
    }
}

pub mod prelude {
    //! Common types used by most callers.
    pub use crate::{CublasError, DeviceBuf, Diag, Handle, Result, Triangular};
    pub use cublas_core::{BlasScalar, GemmConfig, MatrixLayout, Transpose};
}
