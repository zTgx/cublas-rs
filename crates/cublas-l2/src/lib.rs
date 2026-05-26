//! BLAS Level 2 — matrix-vector operations.
//!
//! Each host op takes a typed kernel module and a stream. End users go
//! through `cublas_rs::Handle`, which owns the modules. `Modules::load`
//! is the wiring the facade uses.

use std::sync::Arc;

use cublas_core::Result;
use cuda_core::CudaContext;

mod gemv;
mod symv;
pub mod trsv; // public to expose `Triangular` / `Diag` enums

// Host-slice convenience wrappers.
pub use gemv::{dgemv, hgemv, sgemv, sgemv_tiled};
pub use symv::ssymv;
pub use trsv::strsv;

// Device-buffer primary path.
pub use gemv::{dgemv_dev, sgemv_dev, sgemv_tiled_dev};
pub use symv::ssymv_dev;
pub use trsv::strsv_dev;

/// All L2 kernel modules, typed and ready to launch.
pub struct Modules {
    pub gemv: gemv::kernels::LoadedModule,
    pub dgemv: gemv::dgemv_kernels::LoadedModule,
    pub symv: symv::kernels::LoadedModule,
    pub trsv: trsv::kernels::LoadedModule,
}

impl Modules {
    /// Loads `cublas_l2.ptx` (cwd) and types each kernel view from a single
    /// shared `CudaModule`.
    #[tracing::instrument(level = "debug", skip(ctx))]
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let path = "cublas_l2.ptx";
        tracing::debug!(ptx = path, "loading L2 PTX");
        let raw = ctx.load_module_from_file(path)?;
        Ok(Self {
            gemv: gemv::kernels::from_module(raw.clone())?,
            dgemv: gemv::dgemv_kernels::from_module(raw.clone())?,
            symv: symv::kernels::from_module(raw.clone())?,
            trsv: trsv::kernels::from_module(raw)?,
        })
    }
}
