// SSYMV: y = alpha * A * x + beta * y, A symmetric n × n
//
// Only the specified triangle (Upper or Lower) of A is read; the other
// half is reconstructed by symmetry: A[j,i] := A[i,j].
//
// One thread per row of y. The inner loop hits both stored triangle reads
// (A[i,j] for j ≤ i if Lower) and mirrored reads (A[j,i] for j > i).

use crate::trsv::Triangular;
use cublas_core::Result;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// A is stored Lower-triangular. For j > i we read A[j, i].
    #[kernel]
    pub fn ssymv_lower(
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        let n_size = n as usize;
        if row < n_size {
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < n_size {
                let aij = if j <= row {
                    a[row * n_size + j]
                } else {
                    a[j * n_size + row]
                };
                sum += aij * x[j];
                j += 1;
            }
            if let Some(y_elem) = y.get_mut(idx) {
                *y_elem = alpha * sum + beta * (*y_elem);
            }
        }
    }

    /// A is stored Upper-triangular. For j < i we read A[j, i].
    #[kernel]
    pub fn ssymv_upper(
        n: u32,
        alpha: f32,
        a: &[f32],
        x: &[f32],
        beta: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        let n_size = n as usize;
        if row < n_size {
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < n_size {
                let aij = if j >= row {
                    a[row * n_size + j]
                } else {
                    a[j * n_size + row]
                };
                sum += aij * x[j];
                j += 1;
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
    fields(op = "ssymv", uplo = ?uplo, n),
)]
pub fn ssymv_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    uplo: Triangular,
    n: usize,
    alpha: f32,
    a: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    beta: f32,
    y: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);
    match uplo {
        Triangular::Lower => module.ssymv_lower(stream, cfg, n as u32, alpha, a, x, beta, y)?,
        Triangular::Upper => module.ssymv_upper(stream, cfg, n as u32, alpha, a, x, beta, y)?,
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, x, y),
    fields(op = "ssymv_simple", uplo = ?uplo, n),
)]
pub fn ssymv(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    uplo: Triangular,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> Result<()> {
    assert_eq!(a.len(), n * n, "A length must equal n*n");
    assert!(x.len() >= n, "x is shorter than n");
    assert!(y.len() >= n, "y is shorter than n");
    if n == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let x_dev = DeviceBuffer::from_host(stream, &x[..n])?;
    let mut y_dev = DeviceBuffer::from_host(stream, &y[..n])?;
    ssymv_dev(
        module, stream, uplo, n, alpha, &a_dev, &x_dev, beta, &mut y_dev,
    )?;
    let result = y_dev.to_host_vec(stream)?;
    y[..n].copy_from_slice(&result);
    Ok(())
}
