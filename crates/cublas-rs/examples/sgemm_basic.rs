//! SGEMM end-to-end smoke test (naive variant).
//!
//! NOTE: `sgemm_naive` is currently a `todo!()` stub — this example will
//! panic at runtime. Compiles successfully so it can serve as the L3 usage
//! template until the kernel is filled in.
//!
//! Run with:
//!   cargo oxide run --bin sgemm_basic

use cublas_rs::{prelude::*, sgemm_naive};

fn main() {
    const M: usize = 512;
    const N: usize = 512;
    const K: usize = 512;

    let config = GemmConfig {
        m: M,
        n: N,
        k: K,
        alpha: 1.0f32,
        beta: 0.0f32,
    };

    let a = vec![1.0f32; M * K];
    let b = vec![1.0f32; K * N];
    let mut c = vec![0.0f32; M * N];

    sgemm_naive(&config, &a, &b, &mut c);

    // Each c[i] should equal K (sum of K ones).
    let expected = K as f32;
    let bad = c.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();

    if bad == 0 {
        println!("SGEMM OK: {M}x{N}x{K} verified");
    } else {
        eprintln!("SGEMM FAIL: {bad} mismatches");
        std::process::exit(1);
    }
}
