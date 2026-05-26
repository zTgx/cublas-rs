//! Cross-GPU GEMM benchmark.
//!
//! Runs all SGEMM and DGEMM variants at multiple sizes, reports steady-state
//! GFLOPS. Designed to be portable across GPUs so you can compare e.g.
//! Pascal vs Ampere vs Hopper.
//!
//! Methodology:
//!   1. Allocate device buffers once per size (`Handle::upload`).
//!   2. Warmup `WARMUP_ITERS` iterations to stabilise JIT + caches.
//!   3. Time `TIMED_ITERS` iterations of pure kernel launches.
//!   4. Synchronise the stream before stopping the clock — kernel launches
//!      are async on the host, so without this we'd be measuring launch
//!      overhead only (see `mlp` example for the canonical illustration).
//!   5. Verify correctness on a small (sum-of-ones = K) sanity check.
//!
//! Run with:
//!   cargo oxide run --bin gemm_bench

use std::time::Instant;

use cublas_rs::{DeviceBuf, Handle, prelude::*};

const SIZES: &[usize] = &[256, 512, 1024];
const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;
const TOL: f32 = 1e-3;

fn main() -> Result<()> {
    let h = Handle::new()?;

    println!("GEMM benchmark — cuBLAS-rs");
    println!("  warmup={WARMUP_ITERS}, timed={TIMED_ITERS}, A = B = 1.0 (C[i] should == K)");
    println!();
    println!("            {:>8} {:>10} {:>10} {:>10} {:>10}",
        "size", "naive", "tiled", "vector", "dblbuf");
    println!("  {:─<60}", "");

    // SGEMM section
    for &size in SIZES {
        bench_sgemm(&h, size)?;
    }

    println!();
    println!("DGEMM (Pascal cuts FP64 to ~1/32 SP; Ampere A100 ~1/2):");
    println!("            {:>8} {:>10} {:>10} {:>10} {:>10}",
        "size", "naive", "tiled", "vector", "dblbuf");
    println!("  {:─<60}", "");
    for &size in SIZES {
        bench_dgemm(&h, size)?;
    }

    Ok(())
}

fn bench_sgemm(h: &Handle, size: usize) -> Result<()> {
    let cfg = GemmConfig::<f32> {
        m: size,
        n: size,
        k: size,
        alpha: 1.0,
        beta: 0.0,
    };

    let a_host = vec![1.0f32; size * size];
    let b_host = vec![1.0f32; size * size];
    let a = h.upload(&a_host)?;
    let b = h.upload(&b_host)?;
    let mut c = DeviceBuf::<f32>::zeroed(h.stream(), size * size)?;

    let naive = time_variant(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.sgemm_naive(cfg, a, b, c)
    })?;
    let tiled = time_variant(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.sgemm_tiled(cfg, a, b, c)
    })?;
    let vector = time_variant(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.sgemm_vectorized(cfg, a, b, c)
    })?;
    let dblbuf = time_variant(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.sgemm_double_buf(cfg, a, b, c)
    })?;

    println!(
        "  SGEMM   {size:>8} {naive:>10} {tiled:>10} {vector:>10} {dblbuf:>10}",
    );
    Ok(())
}

fn bench_dgemm(h: &Handle, size: usize) -> Result<()> {
    let cfg = GemmConfig::<f64> {
        m: size,
        n: size,
        k: size,
        alpha: 1.0,
        beta: 0.0,
    };

    let a_host = vec![1.0f64; size * size];
    let b_host = vec![1.0f64; size * size];
    let a = h.upload(&a_host)?;
    let b = h.upload(&b_host)?;
    let mut c = DeviceBuf::<f64>::zeroed(h.stream(), size * size)?;

    let naive = time_variant_d(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.dgemm_naive(cfg, a, b, c)
    })?;
    let tiled = time_variant_d(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.dgemm_tiled(cfg, a, b, c)
    })?;
    let vector = time_variant_d(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.dgemm_vectorized(cfg, a, b, c)
    })?;
    let dblbuf = time_variant_d(h, &cfg, &a, &b, &mut c, size, |h, cfg, a, b, c| {
        h.dgemm_double_buf(cfg, a, b, c)
    })?;

    println!(
        "  DGEMM   {size:>8} {naive:>10} {tiled:>10} {vector:>10} {dblbuf:>10}",
    );
    Ok(())
}

fn time_variant<F>(
    h: &Handle,
    cfg: &GemmConfig<f32>,
    a: &DeviceBuf<f32>,
    b: &DeviceBuf<f32>,
    c: &mut DeviceBuf<f32>,
    size: usize,
    f: F,
) -> Result<String>
where
    F: Fn(&Handle, &GemmConfig<f32>, &DeviceBuf<f32>, &DeviceBuf<f32>, &mut DeviceBuf<f32>) -> Result<()>,
{
    for _ in 0..WARMUP_ITERS {
        f(h, cfg, a, b, c)?;
    }
    h.synchronize()?;

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        f(h, cfg, a, b, c)?;
    }
    h.synchronize()?;
    let elapsed = start.elapsed();

    // Verify last output.
    let c_host = h.download(c)?;
    let expected = size as f32;
    let bad = c_host.iter().filter(|&&v| (v - expected).abs() > TOL).count();
    if bad != 0 {
        return Ok(format!("FAIL ({bad})"));
    }

    let avg = elapsed.as_secs_f64() / TIMED_ITERS as f64;
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = flops / avg / 1e9;
    Ok(format!("{:>6.1}", gflops))
}

fn time_variant_d<F>(
    h: &Handle,
    cfg: &GemmConfig<f64>,
    a: &DeviceBuf<f64>,
    b: &DeviceBuf<f64>,
    c: &mut DeviceBuf<f64>,
    size: usize,
    f: F,
) -> Result<String>
where
    F: Fn(&Handle, &GemmConfig<f64>, &DeviceBuf<f64>, &DeviceBuf<f64>, &mut DeviceBuf<f64>) -> Result<()>,
{
    for _ in 0..WARMUP_ITERS {
        f(h, cfg, a, b, c)?;
    }
    h.synchronize()?;

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        f(h, cfg, a, b, c)?;
    }
    h.synchronize()?;
    let elapsed = start.elapsed();

    let c_host = h.download(c)?;
    let expected = size as f64;
    let bad = c_host
        .iter()
        .filter(|&&v| (v - expected).abs() > TOL as f64)
        .count();
    if bad != 0 {
        return Ok(format!("FAIL ({bad})"));
    }

    let avg = elapsed.as_secs_f64() / TIMED_ITERS as f64;
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = flops / avg / 1e9;
    Ok(format!("{:>6.1}", gflops))
}
