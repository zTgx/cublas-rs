//! BLAS Level 1 — vector operations.
//!
//! All functions take host slices and run a one-shot kernel: allocate device
//! memory, copy in, launch, copy out. See `CLAUDE.md` for the planned v2 API
//! that takes pre-allocated device buffers via a `Handle`.

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
