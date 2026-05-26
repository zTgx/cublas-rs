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
