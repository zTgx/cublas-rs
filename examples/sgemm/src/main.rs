//! SGEMM end-to-end smoke test — runs both the `naive` and `tiled` variants
//! and prints how long each took. Verifies both against the analytical
//! expectation (`C[i] = K` since A and B are all 1.0).
//!
//! Run with:
//!   cargo oxide run --bin sgemm

use std::time::Instant;

use cublas_rs::{Handle, prelude::*};

const M: usize = 512;
const N: usize = 512;
const K: usize = 512;

fn main() {
    let config = GemmConfig {
        m: M,
        n: N,
        k: K,
        alpha: 1.0f32,
        beta: 0.0f32,
    };

    let a = vec![1.0f32; M * K];
    let b = vec![1.0f32; K * N];

    let h = Handle::new().expect("Handle::new");

    println!("SGEMM {}×{}×{} (rows×cols×inner)", M, N, K);

    run_variant("naive", &h, &config, &a, &b, |h, cfg, a, b, c| {
        h.sgemm_naive_simple(cfg, a, b, c)
    });
    run_variant("tiled", &h, &config, &a, &b, |h, cfg, a, b, c| {
        h.sgemm_tiled_simple(cfg, a, b, c)
    });
    run_variant("vector", &h, &config, &a, &b, |h, cfg, a, b, c| {
        h.sgemm_vectorized_simple(cfg, a, b, c)
    });
    run_variant("dblbuf", &h, &config, &a, &b, |h, cfg, a, b, c| {
        h.sgemm_double_buf_simple(cfg, a, b, c)
    });

    // Same drill in f64. Pascal cuts FP64 to ~1/32 of FP32, so expect this to
    // be a lot slower — that's hardware, not the kernel.
    println!();
    println!("DGEMM {}×{}×{} (rows×cols×inner)", M, N, K);
    let cfg_d = GemmConfig::<f64> {
        m: M,
        n: N,
        k: K,
        alpha: 1.0,
        beta: 0.0,
    };
    let a_d = vec![1.0f64; M * K];
    let b_d = vec![1.0f64; K * N];
    run_variant_d("naive", &h, &cfg_d, &a_d, &b_d, |h, cfg, a, b, c| {
        h.dgemm_naive_simple(cfg, a, b, c)
    });
    run_variant_d("tiled", &h, &cfg_d, &a_d, &b_d, |h, cfg, a, b, c| {
        h.dgemm_tiled_simple(cfg, a, b, c)
    });

    // Batched SGEMM: B independent 64×64×64 GEMMs with C[b][i] = K = 64.
    println!();
    const BATCH: usize = 8;
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 64;
    let cfg_b = GemmConfig::<f32> {
        m: BM,
        n: BN,
        k: BK,
        alpha: 1.0,
        beta: 0.0,
    };
    let stride_a = BM * BK;
    let stride_b = BK * BN;
    let stride_c = BM * BN;
    let a_strided = vec![1.0f32; BATCH * stride_a];
    let b_strided = vec![1.0f32; BATCH * stride_b];
    let mut c_strided = vec![0.0f32; BATCH * stride_c];
    let start = Instant::now();
    h.strided_batched_sgemm_simple(
        &cfg_b,
        BATCH,
        &a_strided,
        stride_a,
        &b_strided,
        stride_b,
        &mut c_strided,
        stride_c,
    )
    .expect("strided_batched_sgemm");
    let elapsed = start.elapsed();
    let expected = BK as f32;
    let bad = c_strided
        .iter()
        .filter(|&&v| (v - expected).abs() > 1e-3)
        .count();
    let flops = 2.0 * (BATCH * BM * BN * BK) as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;
    println!("Strided batched SGEMM ({BATCH} × {BM}×{BN}×{BK}):");
    println!(
        "  {:.3} ms ({gflops:.1} GFLOPS) — {}",
        elapsed.as_secs_f64() * 1000.0,
        if bad == 0 { "OK" } else { "FAIL" }
    );
    if bad != 0 {
        std::process::exit(1);
    }

    // batched_sgemm: same workload, per-batch &[f32] slices.
    let a_batches: Vec<Vec<f32>> = (0..BATCH).map(|_| vec![1.0f32; stride_a]).collect();
    let b_batches: Vec<Vec<f32>> = (0..BATCH).map(|_| vec![1.0f32; stride_b]).collect();
    let mut c_batches: Vec<Vec<f32>> = (0..BATCH).map(|_| vec![0.0f32; stride_c]).collect();
    let a_refs: Vec<&[f32]> = a_batches.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f32]> = b_batches.iter().map(|v| v.as_slice()).collect();
    let mut c_refs: Vec<&mut [f32]> = c_batches.iter_mut().map(|v| v.as_mut_slice()).collect();
    h.batched_sgemm(&cfg_b, BATCH, &a_refs, &b_refs, &mut c_refs)
        .expect("batched_sgemm");
    let bad_batched = c_batches
        .iter()
        .flat_map(|v| v.iter())
        .filter(|&&v| (v - expected).abs() > 1e-3)
        .count();
    println!(
        "Batched SGEMM (per-slice API): {}",
        if bad_batched == 0 { "OK" } else { "FAIL" }
    );
    if bad_batched != 0 {
        std::process::exit(1);
    }
}

fn run_variant<F>(name: &str, h: &Handle, cfg: &GemmConfig<f32>, a: &[f32], b: &[f32], f: F)
where
    F: Fn(&Handle, &GemmConfig<f32>, &[f32], &[f32], &mut [f32]) -> Result<()>,
{
    let mut c = vec![0.0f32; M * N];

    // Warmup.
    f(h, cfg, a, b, &mut c).expect("sgemm warmup");

    let start = Instant::now();
    f(h, cfg, a, b, &mut c).expect("sgemm");
    let elapsed = start.elapsed();

    let expected = K as f32;
    let bad = c.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    if bad == 0 {
        println!(
            "  {name:>5}: {:.3} ms  ({:.1} GFLOPS)",
            elapsed.as_secs_f64() * 1000.0,
            gflops
        );
    } else {
        eprintln!("  {name:>5}: FAIL — {bad} mismatches");
        std::process::exit(1);
    }
}

fn run_variant_d<F>(name: &str, h: &Handle, cfg: &GemmConfig<f64>, a: &[f64], b: &[f64], f: F)
where
    F: Fn(&Handle, &GemmConfig<f64>, &[f64], &[f64], &mut [f64]) -> Result<()>,
{
    let mut c = vec![0.0f64; M * N];
    f(h, cfg, a, b, &mut c).expect("dgemm warmup");

    let start = Instant::now();
    f(h, cfg, a, b, &mut c).expect("dgemm");
    let elapsed = start.elapsed();

    let expected = K as f64;
    let bad = c.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    if bad == 0 {
        println!(
            "  {name:>5}: {:.3} ms  ({:.1} GFLOPS)",
            elapsed.as_secs_f64() * 1000.0,
            gflops
        );
    } else {
        eprintln!("  {name:>5}: FAIL — {bad} mismatches");
        std::process::exit(1);
    }
}
