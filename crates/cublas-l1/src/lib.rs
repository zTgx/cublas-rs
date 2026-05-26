//! BLAS Level 1 — vector operations.
//!
//! Each host op takes a typed kernel module and a stream as the first two
//! args. End users don't construct these by hand — they go through
//! `cublas_rs::Handle` which owns the modules and stream and exposes a
//! cuBLAS-style flat API. `Modules::load` is the wiring the facade uses.

use std::sync::Arc;

use cuda_core::{CudaContext, DriverError};

mod asum;
mod axpy;
mod copy;
mod dot;
mod iamax;
mod nrm2;
mod saxpy;
mod scal;

pub use asum::sasum;
pub use axpy::{daxpy, haxpy};
pub use copy::scopy;
pub use dot::dot;
pub use iamax::isamax;
pub use nrm2::nrm2;
pub use saxpy::saxpy;
pub use scal::sscal;

/// All L1 kernel modules, typed and ready to launch. Built once by
/// `cublas_rs::Handle::new()`; not part of the user-facing API surface.
pub struct Modules {
    pub saxpy: saxpy::kernels::LoadedModule,
}

impl Modules {
    /// Loads `cublas_l1.ptx` (must be in cwd — that's where cargo-oxide drops
    /// it during `cargo oxide build`) and types each kernel view.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let raw = ctx.load_module_from_file("cublas_l1.ptx")?;
        Ok(Self {
            saxpy: saxpy::kernels::from_module(raw)?,
        })
    }
}
