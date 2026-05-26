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
