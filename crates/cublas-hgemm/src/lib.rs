pub mod half;
pub mod tensor_core;

// Re-export Tensor Core variant as the default (best performance).
pub use tensor_core::hgemm_tensor_core as hgemm;
