pub mod double_buf;
pub mod naive;
pub mod tiled;
pub mod vectorized;

// Re-export the best-performing variant as the default.
pub use double_buf::*;
pub use naive::*;
pub use tiled::*;
pub use vectorized::*;
