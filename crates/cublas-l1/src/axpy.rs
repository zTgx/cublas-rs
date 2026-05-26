// AXPY: y[i] = alpha * x[i] + y[i] (non-f32 variants)
//
// f32 lives in `saxpy.rs` as the reference template implementation.
//
//   - daxpy — f64 throughout
//   - haxpy — f16 in/out via raw u16 + IEEE-754 bit-twiddle, f32 accumulate
//             (same pattern as hgemv; `half::f16` host intrinsics don't
//             lower in cuda-oxide, so we go through u16 manually)

use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use half::f16;

// ---- DAXPY (f64) -------------------------------------------------------

#[cuda_module]
pub mod daxpy_kernels {
    use super::*;

    #[kernel]
    pub fn daxpy(alpha: f64, x: &[f64], mut y: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(y_elem) = y.get_mut(idx) {
            *y_elem = alpha * x[i] + *y_elem;
        }
    }
}

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "daxpy"))]
pub fn daxpy_dev(
    module: &daxpy_kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f64,
    x: &DeviceBuffer<f64>,
    y: &mut DeviceBuffer<f64>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);
    module.daxpy(stream, cfg, alpha, x, y)?;
    Ok(())
}

#[tracing::instrument(level = "debug", skip(module, stream, x, y), fields(op = "daxpy_simple"))]
pub fn daxpy(
    module: &daxpy_kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f64,
    x: &[f64],
    y: &mut [f64],
) -> Result<()> {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(());
    }
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..n])?;
    daxpy_dev(module, stream, n, alpha, &x_dev, &mut y_dev)?;
    let result = y_dev.to_host_vec(stream)?;
    y[..n].copy_from_slice(&result);
    Ok(())
}

// ---- HAXPY (f16 via u16 bit-twiddle, f32 accumulate) -------------------

#[cuda_module]
pub mod haxpy_kernels {
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

    #[kernel]
    pub fn haxpy(alpha: f32, x: &[u16], mut y: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(y_elem) = y.get_mut(idx) {
            let x_v = f16_to_f32(x[i]);
            let y_v = f16_to_f32(*y_elem);
            *y_elem = f32_to_f16(alpha * x_v + y_v);
        }
    }
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, x, y),
    fields(op = "haxpy_simple"),
)]
pub fn haxpy(
    module: &haxpy_kernels::LoadedModule,
    stream: &CudaStream,
    n: usize,
    alpha: f16,
    x: &[f16],
    y: &mut [f16],
) -> Result<()> {
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(());
    }
    // f16 is repr(transparent) over u16 — safe reinterpret.
    let x_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u16>(), n) };
    let y_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(y.as_ptr().cast::<u16>(), n) };
    let x_dev = DeviceBuffer::from_host(stream, x_u16)?;
    let mut y_dev = DeviceBuffer::from_host(stream, y_u16)?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    module.haxpy(stream, cfg, alpha.to_f32(), &x_dev, &mut y_dev)?;
    let result = y_dev.to_host_vec(stream)?;
    for (i, v) in result.iter().enumerate() {
        y[i] = f16::from_bits(*v);
    }
    Ok(())
}
