// SDOT: sum(x[i] * y[i])
//
// Single-block, 32-thread reduction using shared memory. Correct on every
// arch from Kepler (sm_30) up; *slow* for large n because only one warp
// processes the whole vector.
//
// TODO(perf): upgrade to a multi-block grid-stride design — each block
// computes a partial sum into a per-block buffer, then a second pass (or
// atomicAdd) collapses. Also: on sm_70+ swap the shared-mem tree for a
// `warp::shuffle_xor_f32` butterfly. Skipped on sm_61 because LLVM's nvptx
// backend can't select `llvm.nvvm.shfl.sync.bfly.i32` on Pascal.

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sdot(n: u32, x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        static mut SDATA: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut acc = 0.0f32;
        let mut i = tid;
        while i < n as usize {
            acc += x[i] * y[i];
            i += 32;
        }
        unsafe {
            SDATA[tid] = acc;
        }
        thread::sync_threads();

        let mut stride: usize = 16;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SDATA[tid] += SDATA[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = SDATA[0];
            }
        }
    }
}

const BLOCK_SIZE: u32 = 32;

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "sdot"))]
pub fn sdot_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &DeviceBuffer<f32>,
    y: &DeviceBuffer<f32>,
) -> Result<f32> {
    if n == 0 {
        return Ok(0.0);
    }
    let mut out = DeviceBuffer::<f32>::zeroed(stream, 1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    module.sdot(stream, cfg, n as u32, x, y, &mut out)?;
    let host = out.to_host_vec(stream)?;
    Ok(host[0])
}

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "sdot_simple"))]
pub fn sdot(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &[f32],
    y: &[f32],
) -> Result<f32> {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(0.0);
    }
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    let y_dev = DeviceBuffer::from_host(stream, &y[..n])?;
    sdot_dev(module, stream, n, &x_dev, &y_dev)
}
