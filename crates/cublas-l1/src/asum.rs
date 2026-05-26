// SASUM: sum(|x[i]|)
//
// 32-thread single-block shared-memory tree reduction over |x[i]|. See
// `dot.rs` for the multi-block / warp-shuffle perf TODO.

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sasum(n: u32, x: &[f32], mut out: DisjointSlice<f32>) {
        static mut SDATA: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut acc = 0.0f32;
        let mut i = tid;
        while i < n as usize {
            let v = x[i];
            acc += if v < 0.0 { -v } else { v };
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

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "sasum"))]
pub fn sasum_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &DeviceBuffer<f32>,
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
    module.sasum(stream, cfg, n as u32, x, &mut out)?;
    let host = out.to_host_vec(stream)?;
    Ok(host[0])
}

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "sasum_simple"))]
pub fn sasum(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &[f32],
) -> Result<f32> {
    assert!(x.len() >= n, "x is shorter than n");
    if n == 0 {
        return Ok(0.0);
    }
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    sasum_dev(module, stream, n, &x_dev)
}
