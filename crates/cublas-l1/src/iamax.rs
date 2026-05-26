// ISAMAX: argmax(|x[i]|), 0-based.
//
// 32-thread shared-memory tree reduction tracking (|value|, index) pairs.

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn isamax(n: u32, x: &[f32], mut out: DisjointSlice<u32>) {
        static mut SVAL: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut SIDX: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut best_val = -1.0f32;
        let mut best_idx: u32 = 0;

        let mut i = tid;
        while i < n as usize {
            let v = x[i];
            let av = if v < 0.0 { -v } else { v };
            if av > best_val {
                best_val = av;
                best_idx = i as u32;
            }
            i += 32;
        }

        unsafe {
            SVAL[tid] = best_val;
            SIDX[tid] = best_idx;
        }
        thread::sync_threads();

        let mut stride: usize = 16;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let other_val = SVAL[tid + stride];
                    if other_val > SVAL[tid] {
                        SVAL[tid] = other_val;
                        SIDX[tid] = SIDX[tid + stride];
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = SIDX[0];
            }
        }
    }
}

const BLOCK_SIZE: u32 = 32;

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "isamax"))]
pub fn isamax_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &DeviceBuffer<f32>,
) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    let mut out = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    module.isamax(stream, cfg, n as u32, x, &mut out)?;
    let host = out.to_host_vec(stream)?;
    Ok(host[0] as usize)
}

#[tracing::instrument(level = "debug", skip(module, stream, x), fields(op = "isamax_simple"))]
pub fn isamax(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    x: &[f32],
) -> Result<usize> {
    assert!(x.len() >= n, "x is shorter than n");
    if n == 0 {
        return Ok(0);
    }
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    isamax_dev(module, stream, n, &x_dev)
}
