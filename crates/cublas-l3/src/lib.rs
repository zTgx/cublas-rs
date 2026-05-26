//! BLAS Level 3 — matrix-matrix kernels (plus batched extensions).
//!
//! Organised by precision family, each with its own progression of
//! implementation variants (`naive` → `tiled` → `vectorized` → `double_buf`).
//! End users call through `cublas_rs::Handle`; the free fns and `Modules`
//! below are internal wiring.

use std::sync::Arc;

use cublas_core::Result;
use cuda_core::CudaContext;

pub mod batched;
pub mod dgemm;
pub mod hgemm;
pub mod sgemm;

// Host-slice convenience wrappers.
pub use batched::{batched_sgemm, strided_batched_sgemm};
pub use dgemm::{dgemm_double_buf, dgemm_naive, dgemm_tiled, dgemm_vectorized};
pub use hgemm::{hgemm_half, hgemm_tensor_core};
pub use sgemm::{sgemm_double_buf, sgemm_naive, sgemm_tiled, sgemm_vectorized};

// Device-buffer primary path.
pub use dgemm::{dgemm_naive_dev, dgemm_tiled_dev};
pub use sgemm::{sgemm_naive_dev, sgemm_tiled_dev};

/// All L3 kernel modules, typed and ready to launch. Built once by
/// `cublas_rs::Handle::new()`.
pub struct Modules {
    pub sgemm_naive: sgemm::naive::kernels::LoadedModule,
    pub sgemm_tiled: sgemm::tiled::kernels::LoadedModule,
    pub dgemm_naive: dgemm::naive::kernels::LoadedModule,
    pub dgemm_tiled: dgemm::tiled::kernels::LoadedModule,
}

impl Modules {
    /// Loads `cublas_l3.ptx` (cwd) and types each kernel view from a single
    /// shared `CudaModule`.
    #[tracing::instrument(level = "debug", skip(ctx))]
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let path = "cublas_l3.ptx";
        tracing::debug!(ptx = path, "loading L3 PTX");
        let raw = ctx.load_module_from_file(path)?;
        Ok(Self {
            sgemm_naive: sgemm::naive::kernels::from_module(raw.clone())?,
            sgemm_tiled: sgemm::tiled::kernels::from_module(raw.clone())?,
            dgemm_naive: dgemm::naive::kernels::from_module(raw.clone())?,
            dgemm_tiled: dgemm::tiled::kernels::from_module(raw)?,
        })
    }
}
