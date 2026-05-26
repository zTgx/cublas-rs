//! BLAS Level 3 — matrix-matrix kernels (plus batched extensions).
//!
//! Organised by precision family, each with its own progression of
//! implementation variants (`naive` → `tiled` → `vectorized` → `double_buf`).
//! End users call through `cublas_rs::Handle`; the free fns and `Modules`
//! below are internal wiring.

use std::sync::Arc;

use cuda_core::{CudaContext, DriverError};

pub mod batched;
pub mod dgemm;
pub mod hgemm;
pub mod sgemm;

pub use batched::{batched_sgemm, strided_batched_sgemm};
pub use dgemm::{dgemm_double_buf, dgemm_naive, dgemm_tiled, dgemm_vectorized};
pub use hgemm::{hgemm_half, hgemm_tensor_core};
pub use sgemm::{sgemm_double_buf, sgemm_naive, sgemm_tiled, sgemm_vectorized};

/// All L3 kernel modules, typed and ready to launch. Built once by
/// `cublas_rs::Handle::new()`.
pub struct Modules {
    pub sgemm_naive: sgemm::naive::kernels::LoadedModule,
}

impl Modules {
    /// Loads `cublas_l3.ptx` (cwd) and types each kernel view.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let raw = ctx.load_module_from_file("cublas_l3.ptx")?;
        Ok(Self {
            sgemm_naive: sgemm::naive::kernels::from_module(raw)?,
        })
    }
}
