//! GPU Hello World — minimal toolchain smoke test.
//!
//! Validates the full pipeline end-to-end:
//!   cuda-oxide codegen → PTX → driver → kernel launch → device printf
//!
//! No BLAS dependency, no shared memory, no fancy intrinsics. Just thread
//! indexing and `gpu_printf!` — supported on every CUDA card from sm_20
//! (2010, Fermi) onward.
//!
//! Run with:
//!   cargo oxide run            (default-run)
//!   cargo oxide run --bin hello

use cuda_core::{CudaContext, LaunchConfig};
use cuda_device::{cuda_module, gpu_printf, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Every thread in the block prints its own ID from inside the GPU.
    #[kernel]
    pub fn hello() {
        let tid = thread::index_1d().get();
        gpu_printf!("Hello from GPU thread {}\n", tid);
    }
}

fn main() {
    println!("[host] initializing CUDA...");
    let ctx = CudaContext::new(0).expect("CUDA context init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load hello PTX");

    // 1 block, 8 threads — small enough that the printf output stays readable
    // on any hardware.
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("[host] launching kernel...");
    module.hello(&stream, cfg).expect("kernel launch");
    stream.synchronize().expect("stream sync");

    println!("[host] done — 8 GPU printf lines should appear above.");
}
