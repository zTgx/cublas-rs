pub mod naive;
pub mod tiled;
pub mod double_buf;
pub mod vectorized;

// Re-export the best-performing variant as the default.
pub use tiled::*;
pub use naive::*;
pub use double_buf::*;
pub use vectorized::*;