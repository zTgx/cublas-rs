//! HGEMM — half-precision (f16) matrix-matrix multiply.
//!
//! - [`half`] — pure scalar f16 arithmetic. Works on any GPU.
//! - [`tensor_core`] — WMMA / `mma.sync` path. **Blocked**: cuda-oxide does
//!   not yet expose WMMA intrinsics; the signature is kept so callers can
//!   compile-test against the eventual API.

mod half;
mod tensor_core;

pub use half::hgemm_half;
pub use tensor_core::hgemm_tensor_core;
