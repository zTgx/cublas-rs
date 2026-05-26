// GEMV: y = alpha * op(A) * x + beta * y
//
// A is m × n row-major. `op(A)` is `A` (NoTrans) or `Aᵀ` (Trans).
// NoTrans: x has length n, y has length m, kernel runs one thread per row of y.
// Trans:   x has length m, y has length n, kernel runs one thread per col of A.
//
// Naive variant — single thread per output element, no shared memory. Same
// teaching baseline as `sgemm_naive`. A `_tiled` follow-up would put `x`
// into shared memory (it's reused across every row).

use cublas_core::{Result, Transpose};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use half::f16;

#[cuda_module]
pub mod kernels {
    use super::*;

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

    /// y[j] = alpha * sum_i A[i, j] * x[i] + beta * y[j]   (i.e. y = αAᵀx + βy)
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
}

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

/// DGEMV — double-precision matrix-vector multiply. Stub.
pub fn dgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> Result<()> {
    let _ = (trans, m, n, alpha, a, x, beta, y);
    todo!("launch DGEMV kernel")
}

/// HGEMV — half-precision matrix-vector multiply. Stub.
pub fn hgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f16,
    a: &[f16],
    x: &[f16],
    beta: f16,
    y: &mut [f16],
) -> Result<()> {
    let _ = (trans, m, n, alpha, a, x, beta, y);
    todo!("launch HGEMV kernel")
}
