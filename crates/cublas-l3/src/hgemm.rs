// HGEMM — half-precision (f16) matrix-matrix multiply.
//
// - `hgemm_half`        — scalar f16 arithmetic via raw u16 + IEEE-754
//                         bit-twiddle (same pattern as `hgemv` / `haxpy`).
//                         Tiled (16×16 shared-mem) on top, f32 accumulate.
// - `hgemm_tensor_core` — WMMA / mma.sync. **Blocked**: cuda-oxide doesn't
//                         expose WMMA intrinsics yet. Stub for the API.

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use half::f16;

const TILE_SIZE: usize = 16;

#[cuda_module]
pub mod kernels {
    use super::*;

    fn f16_to_f32(h: u16) -> f32 {
        let h = h as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mantissa = h & 0x3ff;
        if exp == 0 {
            return f32::from_bits(sign);
        }
        if exp == 31 {
            return f32::from_bits(sign | (0xff << 23) | (mantissa << 13));
        }
        f32::from_bits(sign | ((exp + 112) << 23) | (mantissa << 13))
    }

    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        if exp == 0xff {
            let q = if mantissa != 0 { 1 } else { 0 };
            return sign | 0x7c00 | q;
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7c00;
        }
        if new_exp <= 0 {
            return sign;
        }
        sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
    }

    /// 16×16 tiled HGEMM. Tiles loaded as raw u16; converted to f32 at use
    /// time. `alpha` / `beta` are passed in as f32 for accumulator precision.
    #[kernel]
    pub fn hgemm_half(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[u16],
        b: &[u16],
        beta: f32,
        mut c: DisjointSlice<u16, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<u16, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<u16, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE_SIZE);
        let mut sum = 0.0f32;

        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;
            let smem_idx = ty * TILE_SIZE + tx;

            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem_idx] = if row < m_size && a_col < k_size {
                    a[row * k_size + a_col]
                } else {
                    0
                };
                let b_row = tile_start + ty;
                TILE_B[smem_idx] = if b_row < k_size && col < n_size {
                    b[b_row * n_size + col]
                } else {
                    0
                };
            }

            thread::sync_threads();

            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    let a_v = f16_to_f32(TILE_A[ty * TILE_SIZE + i]);
                    let b_v = f16_to_f32(TILE_B[i * TILE_SIZE + tx]);
                    sum += a_v * b_v;
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    let cur = f16_to_f32(*c_elem);
                    *c_elem = f32_to_f16(alpha * sum + beta * cur);
                }
            }
        }
    }
}

/// Tiled HGEMM. Host slices only — `DeviceBuffer<f16>` ↔ raw u16 interop is
/// awkward; the kernel works on raw u16 internally.
#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "hgemm_half", m = config.m, n = config.n, k = config.k),
)]
pub fn hgemm_half(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f16>,
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    assert_eq!(a.len(), m * k, "A length must equal m*k");
    assert_eq!(b.len(), k * n, "B length must equal k*n");
    assert_eq!(c.len(), m * n, "C length must equal m*n");
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    let a_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u16>(), a.len()) };
    let b_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u16>(), b.len()) };
    let c_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(c.as_ptr().cast::<u16>(), c.len()) };

    let a_dev = DeviceBuffer::from_host(stream, a_u16)?;
    let b_dev = DeviceBuffer::from_host(stream, b_u16)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c_u16)?;

    let tile = TILE_SIZE as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch HGEMM half");
    module.hgemm_half(
        stream,
        cfg,
        m as u32,
        n as u32,
        k as u32,
        alpha.to_f32(),
        &a_dev,
        &b_dev,
        beta.to_f32(),
        &mut c_dev,
    )?;

    let result = c_dev.to_host_vec(stream)?;
    for (i, v) in result.iter().enumerate() {
        c[i] = f16::from_bits(*v);
    }
    Ok(())
}

/// Tensor-Core HGEMM kernel launch (f16, sm_80+). Blocked on WMMA wrapper.
pub fn hgemm_tensor_core(config: &GemmConfig<f16>, a: &[f16], b: &[f16], c: &mut [f16]) -> Result<()> {
    let _ = (config, a, b, c);
    todo!("blocked on WMMA wrapper in cuda-oxide")
}
