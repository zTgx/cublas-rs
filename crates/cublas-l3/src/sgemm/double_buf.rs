// SGEMM double buffer: manual two-tile shared memory pipeline
//
// Note: real cp.async-based pipelining (Ampere) blocked until cuda-oxide
// exposes the non-bulk cp.async intrinsic. This variant uses two SharedArray
// buffers with explicit sync_threads.

use cublas_core::GemmConfig;

/// Double-buffered SGEMM kernel launch.
pub fn sgemm_double_buf(config: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
    let _ = (config, a, b, c);
    todo!("launch double-buffered SGEMM kernel")
}
