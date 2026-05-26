// GEMV: y = alpha * op(A) * x + beta * y
//
// A is m × n row-major. `op(A)` is `A` (NoTrans) or `Aᵀ` (Trans).
// NoTrans: x has length n, y has length m.
// Trans:   x has length m, y has length n.
//
// Two variants:
//   - `*_naive`: one thread per output element, no shared memory. Baseline.
//   - `*_tiled`: one block (32 threads) per output element. Block cooperatively
//     loads tiles of `x` into shared memory; the 32 lanes accumulate
//     partials and shared-memory-reduce. Cuts global reads of `x` by ~32×
//     (each x[j] is loaded once per block, not n times like the naive
//     L1-fed version).

use cublas_core::{Result, Transpose};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use half::f16;

const TILE: usize = 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    // ---- Naive: one thread per output element ----------------------------

    /// y[i] = alpha * sum_j A[i, j] * x[j] + beta * y[i]
    #[kernel]
    pub fn sgemv_n(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row < m as usize {
            let n_size = n as usize;
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < n_size {
                sum += a[row * n_size + j] * x[j];
                j += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                *y_elem = alpha * sum + beta * (*y_elem);
            }
        }
    }

    /// y[j] = alpha * sum_i A[i, j] * x[i] + beta * y[j]
    #[kernel]
    pub fn sgemv_t(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let col = idx.get();
        if col < n as usize {
            let n_size = n as usize;
            let m_size = m as usize;
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i < m_size {
                sum += a[i * n_size + col] * x[i];
                i += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                *y_elem = alpha * sum + beta * (*y_elem);
            }
        }
    }

    // ---- Tiled: one block per output element, x cached in shared --------

    /// NoTrans tiled. block_dim = (32, 1, 1); grid_dim = (m, 1, 1).
    /// Each block computes one y[row]. Loops over `x` in 32-wide tiles, each
    /// tile loaded cooperatively into shared memory then dotted with the
    /// matching A column-strip.
    #[kernel]
    pub fn sgemv_n_tiled(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut SX: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut SUM: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let n_size = n as usize;
        let m_size = m as usize;

        if row >= m_size {
            return;
        }

        let num_tiles = n_size.div_ceil(TILE);
        let mut acc = 0.0f32;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE;
            let j = tile_start + tid;
            unsafe {
                SX[tid] = if j < n_size { x[j] } else { 0.0 };
            }
            thread::sync_threads();

            if j < n_size {
                unsafe {
                    acc += a[row * n_size + j] * SX[tid];
                }
            }
            thread::sync_threads();
            tile += 1;
        }

        unsafe {
            SUM[tid] = acc;
        }
        thread::sync_threads();

        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                let val = SUM[0];
                let y_elem = y.get_unchecked_mut(row);
                *y_elem = alpha * val + beta * (*y_elem);
            }
        }
    }

    /// Trans tiled. block_dim = (32, 1, 1); grid_dim = (n, 1, 1).
    /// Each block computes one y[col]. Tiles over `x` (length m) in shared.
    #[kernel]
    pub fn sgemv_t_tiled(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut SX: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut SUM: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let col = thread::blockIdx_x() as usize;
        let n_size = n as usize;
        let m_size = m as usize;

        if col >= n_size {
            return;
        }

        let num_tiles = m_size.div_ceil(TILE);
        let mut acc = 0.0f32;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE;
            let i = tile_start + tid;
            unsafe {
                SX[tid] = if i < m_size { x[i] } else { 0.0 };
            }
            thread::sync_threads();

            if i < m_size {
                unsafe {
                    acc += a[i * n_size + col] * SX[tid];
                }
            }
            thread::sync_threads();
            tile += 1;
        }

        unsafe {
            SUM[tid] = acc;
        }
        thread::sync_threads();

        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                let val = SUM[0];
                let y_elem = y.get_unchecked_mut(col);
                *y_elem = alpha * val + beta * (*y_elem);
            }
        }
    }
}

// ---- Naive launchers ----------------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "sgemv", trans = ?trans, m, n),
)]
pub fn sgemv_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    beta: f32,
    y: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    match trans {
        Transpose::NoTrans => {
            let cfg = LaunchConfig::for_num_elems(m as u32);
            module.sgemv_n(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
        }
        Transpose::Trans => {
            let cfg = LaunchConfig::for_num_elems(n as u32);
            module.sgemv_t(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
        }
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "sgemv_simple", trans = ?trans, m, n),
)]
pub fn sgemv(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> Result<()> {
    assert_eq!(a.len(), m * n, "A length must equal m*n");
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans => (m, n),
    };
    assert!(x.len() >= x_len, "x is shorter than expected");
    assert!(y.len() >= y_len, "y is shorter than expected");
    if m == 0 || n == 0 {
        return Ok(());
    }

    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let x_dev = DeviceBuffer::from_host(stream, &x[..x_len])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..y_len])?;
    sgemv_dev(
        module, stream, trans, m, n, alpha, &a_dev, &x_dev, beta, &mut y_dev,
    )?;
    let result = y_dev.to_host_vec(stream)?;
    y[..y_len].copy_from_slice(&result);
    Ok(())
}

// ---- Tiled launchers ----------------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "sgemv_tiled", trans = ?trans, m, n),
)]
pub fn sgemv_tiled_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    beta: f32,
    y: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    let (grid, kernel_fn): (u32, _) = match trans {
        Transpose::NoTrans => (m as u32, 0),
        Transpose::Trans => (n as u32, 1),
    };
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    if kernel_fn == 0 {
        module.sgemv_n_tiled(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
    } else {
        module.sgemv_t_tiled(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "sgemv_tiled_simple", trans = ?trans, m, n),
)]
pub fn sgemv_tiled(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> Result<()> {
    assert_eq!(a.len(), m * n, "A length must equal m*n");
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans => (m, n),
    };
    assert!(x.len() >= x_len, "x is shorter than expected");
    assert!(y.len() >= y_len, "y is shorter than expected");
    if m == 0 || n == 0 {
        return Ok(());
    }

    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let x_dev = DeviceBuffer::from_host(stream, &x[..x_len])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..y_len])?;
    sgemv_tiled_dev(
        module, stream, trans, m, n, alpha, &a_dev, &x_dev, beta, &mut y_dev,
    )?;
    let result = y_dev.to_host_vec(stream)?;
    y[..y_len].copy_from_slice(&result);
    Ok(())
}

// ---- Stubs --------------------------------------------------------------

// ---- DGEMV (f64) --------------------------------------------------------

#[cuda_module]
pub mod dgemv_kernels {
    use super::*;

    #[kernel]
    pub fn dgemv_n(
        m: u32,
        n: u32,
        alpha: f64,
        a: &[f64],
        x: &[f64],
        beta: f64,
        mut y: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row < m as usize {
            let n_size = n as usize;
            let mut sum = 0.0f64;
            let mut j = 0usize;
            while j < n_size {
                sum += a[row * n_size + j] * x[j];
                j += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                *y_elem = alpha * sum + beta * (*y_elem);
            }
        }
    }

    #[kernel]
    pub fn dgemv_t(
        m: u32,
        n: u32,
        alpha: f64,
        a: &[f64],
        x: &[f64],
        beta: f64,
        mut y: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let col = idx.get();
        if col < n as usize {
            let n_size = n as usize;
            let m_size = m as usize;
            let mut sum = 0.0f64;
            let mut i = 0usize;
            while i < m_size {
                sum += a[i * n_size + col] * x[i];
                i += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                *y_elem = alpha * sum + beta * (*y_elem);
            }
        }
    }
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "dgemv", trans = ?trans, m, n),
)]
pub fn dgemv_dev(
    module: &dgemv_kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &DeviceBuffer<f64>,
    x: &DeviceBuffer<f64>,
    beta: f64,
    y: &mut DeviceBuffer<f64>,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    match trans {
        Transpose::NoTrans => {
            let cfg = LaunchConfig::for_num_elems(m as u32);
            module.dgemv_n(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
        }
        Transpose::Trans => {
            let cfg = LaunchConfig::for_num_elems(n as u32);
            module.dgemv_t(stream, cfg, m as u32, n as u32, alpha, a, x, beta, y)?;
        }
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "dgemv_simple", trans = ?trans, m, n),
)]
pub fn dgemv(
    module: &dgemv_kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> Result<()> {
    assert_eq!(a.len(), m * n, "A length must equal m*n");
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans => (m, n),
    };
    assert!(x.len() >= x_len, "x is shorter than expected");
    assert!(y.len() >= y_len, "y is shorter than expected");
    if m == 0 || n == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let x_dev = DeviceBuffer::from_host(stream, &x[..x_len])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..y_len])?;
    dgemv_dev(
        module, stream, trans, m, n, alpha, &a_dev, &x_dev, beta, &mut y_dev,
    )?;
    let result = y_dev.to_host_vec(stream)?;
    y[..y_len].copy_from_slice(&result);
    Ok(())
}

// ---- HGEMV (f16 in/out via raw u16, f32 accumulate) -------------------

#[cuda_module]
pub mod hgemv_kernels {
    use super::*;

    // IEEE-754 binary16 → binary32. Subnormals flush to zero.
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

    // IEEE-754 binary32 → binary16. Underflow flushes to zero, overflow
    // saturates to infinity.
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
    pub fn hgemv_n(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[u16],
        x: &[u16],
        beta: f32,
        mut y: DisjointSlice<u16>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row < m as usize {
            let n_size = n as usize;
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < n_size {
                sum += f16_to_f32(a[row * n_size + j]) * f16_to_f32(x[j]);
                j += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                let cur = f16_to_f32(*y_elem);
                *y_elem = f32_to_f16(alpha * sum + beta * cur);
            }
        }
    }

    #[kernel]
    pub fn hgemv_t(
        m: u32,
        n: u32,
        alpha: f32,
        a: &[u16],
        x: &[u16],
        beta: f32,
        mut y: DisjointSlice<u16>,
    ) {
        let idx = thread::index_1d();
        let col = idx.get();
        if col < n as usize {
            let n_size = n as usize;
            let m_size = m as usize;
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i < m_size {
                sum += f16_to_f32(a[i * n_size + col]) * f16_to_f32(x[i]);
                i += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                let cur = f16_to_f32(*y_elem);
                *y_elem = f32_to_f16(alpha * sum + beta * cur);
            }
        }
    }
}

/// HGEMV host-slice path. Reinterprets `&[f16]` as `&[u16]` to feed kernels
/// that do the conversion in bit-arithmetic; `half::f16` is
/// `#[repr(transparent)]` over `u16`, so the cast is sound.
#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "hgemv", trans = ?trans, m, n),
)]
pub fn hgemv(
    module: &hgemv_kernels::LoadedModule,
    stream: &CudaStream,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f16,
    a: &[f16],
    x: &[f16],
    beta: f16,
    y: &mut [f16],
) -> Result<()> {
    assert_eq!(a.len(), m * n, "A length must equal m*n");
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans => (m, n),
    };
    assert!(x.len() >= x_len, "x is shorter than expected");
    assert!(y.len() >= y_len, "y is shorter than expected");
    if m == 0 || n == 0 {
        return Ok(());
    }

    let a_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u16>(), a.len()) };
    let x_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u16>(), x_len) };
    let y_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(y.as_ptr().cast::<u16>(), y_len) };

    let a_dev = DeviceBuffer::from_host(stream, a_u16)?;
    let x_dev = DeviceBuffer::from_host(stream, x_u16)?;
    let mut y_dev = DeviceBuffer::from_host(stream, y_u16)?;

    let alpha32 = alpha.to_f32();
    let beta32 = beta.to_f32();
    match trans {
        Transpose::NoTrans => {
            let cfg = LaunchConfig::for_num_elems(m as u32);
            module.hgemv_n(
                stream, cfg, m as u32, n as u32, alpha32, &a_dev, &x_dev, beta32, &mut y_dev,
            )?;
        }
        Transpose::Trans => {
            let cfg = LaunchConfig::for_num_elems(n as u32);
            module.hgemv_t(
                stream, cfg, m as u32, n as u32, alpha32, &a_dev, &x_dev, beta32, &mut y_dev,
            )?;
        }
    }

    let result = y_dev.to_host_vec(stream)?;
    for (i, v) in result.iter().enumerate() {
        y[i] = f16::from_bits(*v);
    }
    Ok(())
}
