// TRSV: solve op(A) * x = b in place (x overwrites b)
//
// A is n × n triangular (row-major). Gates Cholesky / LU solvers.
//
// Triangular-solve is fundamentally sequential — each x[i] depends on
// x[0..i] (Lower NoTrans) or x[i+1..n] (Upper NoTrans). A truly parallel
// version uses level scheduling or column-major tile decomposition; for
// v1 we run a single-thread kernel. Slow but correct. Marker for a future
// `strsv_tiled` perf upgrade.
//
// TODO(perf): block-row decomposition. Solve the n×TILE diagonal block
// sequentially (still single-thread), then sgemv-update the trailing
// block in parallel. Iterate TILE rows at a time.

use cublas_core::{Result, Transpose};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

/// Whether A is upper- or lower-triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Triangular {
    Upper,
    Lower,
}

/// Whether the triangular matrix has a unit (implicit 1) diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    NonUnit,
    Unit,
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Solve L x = b, L lower-triangular. Single-thread sequential.
    #[kernel]
    pub fn strsv_lower_n(n: u32, unit_diag: u32, a: &[f32], mut x: DisjointSlice<f32>) {
        let n_size = n as usize;
        let mut i = 0usize;
        while i < n_size {
            let mut sum = unsafe { *x.get_unchecked_mut(i) };
            let mut j = 0usize;
            while j < i {
                sum -= a[i * n_size + j] * unsafe { *x.get_unchecked_mut(j) };
                j += 1;
            }
            let v = if unit_diag == 1 { sum } else { sum / a[i * n_size + i] };
            unsafe {
                *x.get_unchecked_mut(i) = v;
            }
            i += 1;
        }
    }

    /// Solve U x = b, U upper-triangular. Single-thread back-substitution.
    #[kernel]
    pub fn strsv_upper_n(n: u32, unit_diag: u32, a: &[f32], mut x: DisjointSlice<f32>) {
        let n_size = n as usize;
        let mut i = n_size;
        while i > 0 {
            i -= 1;
            let mut sum = unsafe { *x.get_unchecked_mut(i) };
            let mut j = i + 1;
            while j < n_size {
                sum -= a[i * n_size + j] * unsafe { *x.get_unchecked_mut(j) };
                j += 1;
            }
            let v = if unit_diag == 1 { sum } else { sum / a[i * n_size + i] };
            unsafe {
                *x.get_unchecked_mut(i) = v;
            }
        }
    }

    /// Solve Lᵀ x = b (= U x = b read from L's columns).
    #[kernel]
    pub fn strsv_lower_t(n: u32, unit_diag: u32, a: &[f32], mut x: DisjointSlice<f32>) {
        let n_size = n as usize;
        let mut i = n_size;
        while i > 0 {
            i -= 1;
            let mut sum = unsafe { *x.get_unchecked_mut(i) };
            let mut j = i + 1;
            while j < n_size {
                sum -= a[j * n_size + i] * unsafe { *x.get_unchecked_mut(j) };
                j += 1;
            }
            let v = if unit_diag == 1 { sum } else { sum / a[i * n_size + i] };
            unsafe {
                *x.get_unchecked_mut(i) = v;
            }
        }
    }

    /// Solve Uᵀ x = b (= L x = b read from U's columns).
    #[kernel]
    pub fn strsv_upper_t(n: u32, unit_diag: u32, a: &[f32], mut x: DisjointSlice<f32>) {
        let n_size = n as usize;
        let mut i = 0usize;
        while i < n_size {
            let mut sum = unsafe { *x.get_unchecked_mut(i) };
            let mut j = 0usize;
            while j < i {
                sum -= a[j * n_size + i] * unsafe { *x.get_unchecked_mut(j) };
                j += 1;
            }
            let v = if unit_diag == 1 { sum } else { sum / a[i * n_size + i] };
            unsafe {
                *x.get_unchecked_mut(i) = v;
            }
            i += 1;
        }
    }
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, b),
    fields(op = "strsv", uplo = ?uplo, trans = ?trans, diag = ?diag, n),
)]
pub fn strsv_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    uplo: Triangular,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &DeviceBuffer<f32>,
    b: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let unit = if diag == Diag::Unit { 1u32 } else { 0u32 };
    match (uplo, trans) {
        (Triangular::Lower, Transpose::NoTrans) => {
            module.strsv_lower_n(stream, cfg, n as u32, unit, a, b)?
        }
        (Triangular::Upper, Transpose::NoTrans) => {
            module.strsv_upper_n(stream, cfg, n as u32, unit, a, b)?
        }
        (Triangular::Lower, Transpose::Trans) => {
            module.strsv_lower_t(stream, cfg, n as u32, unit, a, b)?
        }
        (Triangular::Upper, Transpose::Trans) => {
            module.strsv_upper_t(stream, cfg, n as u32, unit, a, b)?
        }
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, a, b),
    fields(op = "strsv_simple", uplo = ?uplo, trans = ?trans, diag = ?diag, n),
)]
pub fn strsv(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    uplo: Triangular,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &[f32],
    b: &mut [f32],
) -> Result<()> {
    assert_eq!(a.len(), n * n, "A length must equal n*n");
    assert!(b.len() >= n, "b is shorter than n");
    if n == 0 {
        return Ok(());
    }
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let mut b_dev = DeviceBuffer::from_host(stream, &b[..n])?;
    strsv_dev(module, stream, uplo, trans, diag, n, &a_dev, &mut b_dev)?;
    let result = b_dev.to_host_vec(stream)?;
    b[..n].copy_from_slice(&result);
    Ok(())
}
