//! BLAS Level 3 — matrix-matrix kernels (plus batched extensions).
//!
//! Each precision family lives in one flat file:
//!   - `sgemm.rs` — f32, four variants (naive + tiled implemented; vectorized
//!     and double_buf still stubs)
//!   - `dgemm.rs` — f64, same shape
//!   - `hgemm.rs` — f16 (stub for now)
//!   - `batched.rs` — batched / strided-batched (stubs)
//!
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
pub use batched::strided_batched_sgemm_dev;
pub use dgemm::{dgemm_naive_dev, dgemm_tiled_dev};
pub use sgemm::{sgemm_naive_dev, sgemm_tiled_dev};

/// All L3 kernel modules, typed and ready to launch. Built once by
/// `cublas_rs::Handle::new()`.
pub struct Modules {
    pub sgemm: sgemm::kernels::LoadedModule,
    pub dgemm: dgemm::kernels::LoadedModule,
    pub batched: batched::kernels::LoadedModule,
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
            sgemm: sgemm::kernels::from_module(raw.clone())?,
            dgemm: dgemm::kernels::from_module(raw.clone())?,
            batched: batched::kernels::from_module(raw)?,
        })
    }
}
