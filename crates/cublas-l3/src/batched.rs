// Batched SGEMM — many matrix multiplies sharing one kernel launch.
//
// The real kernel is `strided_batched_sgemm`: a 3D grid where blockIdx.z
// picks the batch and each block computes one 16×16 output tile (same
// shape as `sgemm_tiled`). The "array of slices" `batched_sgemm` is a
// thin host wrapper that concatenates the per-batch inputs into one
// contiguous buffer and forwards to the strided kernel.
//
// Constraint: all batches must share `m × n × k`. cuBLAS's variable-shape
// `cublasSgemmGroupedBatched` is not modelled here — would need either a
// device-pointer table or per-batch streams; punt to a follow-up.

use cublas_core::{GemmConfig, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const TILE_SIZE: usize = 16;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One 16×16 tile of one batch per CUDA block.
    /// `blockIdx.z` selects the batch, `blockIdx.{y,x}` selects the tile.
    #[kernel]
    pub fn strided_batched_sgemm(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        stride_a: u32,
        b: &[f32],
        stride_b: u32,
        beta: f32,
        mut c: DisjointSlice<f32>,
        stride_c: u32,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;
        let batch = thread::blockIdx_z() as usize;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;
        let sa = stride_a as usize;
        let sb = stride_b as usize;
        let sc = stride_c as usize;

        let a_base = batch * sa;
        let b_base = batch * sb;
        let c_base = batch * sc;

        let num_tiles = k_size.div_ceil(TILE_SIZE);
        let mut sum = 0.0f32;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;
            let smem_idx = ty * TILE_SIZE + tx;

            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem_idx] = if row < m_size && a_col < k_size {
                    a[a_base + row * k_size + a_col]
                } else {
                    0.0
                };
                let b_row = tile_start + ty;
                TILE_B[smem_idx] = if b_row < k_size && col < n_size {
                    b[b_base + b_row * n_size + col]
                } else {
                    0.0
                };
            }

            thread::sync_threads();

            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    sum += TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        if row < m_size && col < n_size {
            let c_idx = c_base + row * n_size + col;
            unsafe {
                let cur = *c.get_unchecked_mut(c_idx);
                *c.get_unchecked_mut(c_idx) = alpha * sum + beta * cur;
            }
        }
    }
}

// ---- strided_batched_sgemm ---------------------------------------------

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "strided_batched_sgemm", m = config.m, n = config.n, k = config.k, batch_count),
)]
pub fn strided_batched_sgemm_dev(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &DeviceBuffer<f32>,
    stride_a: usize,
    b: &DeviceBuffer<f32>,
    stride_b: usize,
    c: &mut DeviceBuffer<f32>,
    stride_c: usize,
) -> Result<()> {
    let GemmConfig { m, n, k, alpha, beta } = *config;
    if m == 0 || n == 0 || k == 0 || batch_count == 0 {
        return Ok(());
    }
    let tile = TILE_SIZE as u32;
    let cfg = LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(tile),
            (m as u32).div_ceil(tile),
            batch_count as u32,
        ),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    };
    tracing::trace!(grid = ?cfg.grid_dim, block = ?cfg.block_dim, "launch strided_batched_sgemm");
    module.strided_batched_sgemm(
        stream,
        cfg,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        a,
        stride_a as u32,
        b,
        stride_b as u32,
        beta,
        c,
        stride_c as u32,
    )?;
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "strided_batched_sgemm_simple", m = config.m, n = config.n, k = config.k, batch_count),
)]
pub fn strided_batched_sgemm(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &[f32],
    stride_a: usize,
    b: &[f32],
    stride_b: usize,
    c: &mut [f32],
    stride_c: usize,
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert!(a.len() >= batch_count * stride_a, "A buffer too small for strides");
    assert!(b.len() >= batch_count * stride_b, "B buffer too small for strides");
    assert!(c.len() >= batch_count * stride_c, "C buffer too small for strides");
    if m == 0 || n == 0 || k == 0 || batch_count == 0 {
        return Ok(());
    }

    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, c)?;
    strided_batched_sgemm_dev(
        module,
        stream,
        config,
        batch_count,
        &a_dev,
        stride_a,
        &b_dev,
        stride_b,
        &mut c_dev,
        stride_c,
    )?;
    let result = c_dev.to_host_vec(stream)?;
    c.copy_from_slice(&result);
    Ok(())
}

// ---- batched_sgemm (concatenate + delegate) ---------------------------

/// Same-shape batched SGEMM with per-batch host slices. Concatenates into
/// contiguous storage and forwards to `strided_batched_sgemm`.
#[tracing::instrument(
    level = "debug",
    skip(module, stream, config, a, b, c),
    fields(op = "batched_sgemm", m = config.m, n = config.n, k = config.k, batch_count),
)]
pub fn batched_sgemm(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: &GemmConfig<f32>,
    batch_count: usize,
    a: &[&[f32]],
    b: &[&[f32]],
    c: &mut [&mut [f32]],
) -> Result<()> {
    let GemmConfig { m, n, k, .. } = *config;
    assert_eq!(a.len(), batch_count, "a slices count must equal batch_count");
    assert_eq!(b.len(), batch_count, "b slices count must equal batch_count");
    assert_eq!(c.len(), batch_count, "c slices count must equal batch_count");
    if m == 0 || n == 0 || k == 0 || batch_count == 0 {
        return Ok(());
    }

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let mut a_cat = Vec::with_capacity(batch_count * stride_a);
    let mut b_cat = Vec::with_capacity(batch_count * stride_b);
    let mut c_cat = vec![0.0f32; batch_count * stride_c];
    for batch in 0..batch_count {
        assert_eq!(a[batch].len(), stride_a, "a[{}] wrong size", batch);
        assert_eq!(b[batch].len(), stride_b, "b[{}] wrong size", batch);
        assert_eq!(c[batch].len(), stride_c, "c[{}] wrong size", batch);
        a_cat.extend_from_slice(a[batch]);
        b_cat.extend_from_slice(b[batch]);
        c_cat[batch * stride_c..(batch + 1) * stride_c].copy_from_slice(c[batch]);
    }

    strided_batched_sgemm(
        module,
        stream,
        config,
        batch_count,
        &a_cat,
        stride_a,
        &b_cat,
        stride_b,
        &mut c_cat,
        stride_c,
    )?;

    for batch in 0..batch_count {
        c[batch].copy_from_slice(&c_cat[batch * stride_c..(batch + 1) * stride_c]);
    }
    Ok(())
}
