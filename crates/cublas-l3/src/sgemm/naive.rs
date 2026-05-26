// SGEMM naive: C = alpha * A * B + beta * C
//
// Row-major. One thread per element of C. No shared memory — baseline for
// correctness and bandwidth. Reference for the L3 host-fn template; the
// tiled/vectorized/double-buffered variants improve on top of this.

use cublas_core::GemmConfig;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Naive GEMM: one thread per output element. A is M×K, B is K×N,
    /// C is M×N (all row-major).
    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < m as usize {
                let n_size = n as usize;
                let k_size = k as usize;

                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_size {
                    sum += a[row * k_size + i] * b[i * n_size + col];
                    i += 1;
                }

                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }
}

const BLOCK_SIZE: u32 = 16;

/// Internal launcher. End users go through `cublas_rs::Handle::sgemm_naive`.
pub fn sgemm_naive(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let GemmConfig {
        m,
        n,
        k,
        alpha,
        beta,
    } = *config;
    assert_eq!(a.len(), m * k, "A length must equal m*k");
    assert_eq!(b.len(), k * n, "B length must equal k*n");
    assert_eq!(c.len(), m * n, "C length must equal m*n");

    let a_dev = DeviceBuffer::from_host(stream, a).expect("copy A to device");
    let b_dev = DeviceBuffer::from_host(stream, b).expect("copy B to device");
    let mut c_dev = DeviceBuffer::from_host(stream, c).expect("copy C to device");

    let grid_x = (n as u32).div_ceil(BLOCK_SIZE);
    let grid_y = (m as u32).div_ceil(BLOCK_SIZE);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        shared_mem_bytes: 0,
    };

    module
        .sgemm_naive(
            stream,
            cfg,
            m as u32,
            n as u32,
            k as u32,
            alpha,
            &a_dev,
            &b_dev,
            beta,
            &mut c_dev,
        )
        .expect("SGEMM naive launch");

    let result = c_dev.to_host_vec(stream).expect("copy C back");
    c.copy_from_slice(&result);
}
