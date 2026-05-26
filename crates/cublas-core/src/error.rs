use cuda_core::DriverError;
use cuda_core::embedded::EmbeddedModuleError;
use thiserror::Error;

/// Unified error returned by cuBLAS-rs ops. Wraps the underlying cuda-oxide
/// errors so callers only need to depend on `cublas-rs`.
#[derive(Debug, Error)]
pub enum CublasError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("failed to load embedded CUDA module: {0}")]
    EmbeddedModule(#[from] EmbeddedModuleError),

    #[error("dimension mismatch in {what}: expected {expected}, got {got}")]
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
}

/// Shorthand for results returned by cuBLAS-rs ops.
pub type Result<T> = std::result::Result<T, CublasError>;
