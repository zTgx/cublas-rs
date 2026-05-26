//! BLAS Level 1 — vector operations.
//!
//! Each host op takes a typed kernel module and a stream as the first two
//! args, plus device or host buffers. End users don't construct these by
//! hand — they go through `cublas_rs::Handle`, which owns the modules and
//! stream and exposes the cuBLAS-style flat API. `Modules::load` is the
//! wiring the facade uses.

use std::sync::Arc;

use cublas_core::Result;
use cuda_core::CudaContext;

mod asum;
mod axpy;
mod copy;
mod dot;
mod iamax;
mod nrm2;
mod saxpy;
mod scal;

// Host-slice convenience wrappers.
pub use asum::sasum;
pub use axpy::{daxpy, haxpy};
pub use copy::scopy;
pub use dot::sdot;
pub use iamax::isamax;
pub use nrm2::snrm2;
pub use saxpy::saxpy;
pub use scal::sscal;

// Device-buffer primary path.
pub use asum::sasum_dev;
pub use copy::scopy_dev;
pub use dot::sdot_dev;
pub use iamax::isamax_dev;
pub use nrm2::snrm2_dev;
pub use saxpy::saxpy_dev;
pub use scal::sscal_dev;

/// All L1 kernel modules, typed and ready to launch. Built once by
/// `cublas_rs::Handle::new()`; not part of the user-facing API surface.
pub struct Modules {
    pub saxpy: saxpy::kernels::LoadedModule,
    pub sscal: scal::kernels::LoadedModule,
    pub scopy: copy::kernels::LoadedModule,
    pub sdot: dot::kernels::LoadedModule,
    pub snrm2: nrm2::kernels::LoadedModule,
    pub sasum: asum::kernels::LoadedModule,
    pub isamax: iamax::kernels::LoadedModule,
}

impl Modules {
    /// Loads `cublas_l1.ptx` (must be in cwd — that's where cargo-oxide drops
    /// it during `cargo oxide build`) and types each kernel view from a
    /// single shared `CudaModule`.
    #[tracing::instrument(level = "debug", skip(ctx))]
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let path = "cublas_l1.ptx";
        tracing::debug!(ptx = path, "loading L1 PTX");
        let raw = ctx.load_module_from_file(path)?;
        Ok(Self {
            saxpy: saxpy::kernels::from_module(raw.clone())?,
            sscal: scal::kernels::from_module(raw.clone())?,
            scopy: copy::kernels::from_module(raw.clone())?,
            sdot: dot::kernels::from_module(raw.clone())?,
            snrm2: nrm2::kernels::from_module(raw.clone())?,
            sasum: asum::kernels::from_module(raw.clone())?,
            isamax: iamax::kernels::from_module(raw)?,
        })
    }
}
