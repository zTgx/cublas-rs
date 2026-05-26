//! SGEMM end-to-end smoke test (naive variant).
//!
//! Run with:
//!   cargo oxide run --bin sgemm

use cublas_rs::{Handle, prelude::*};

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

    let h = Handle::new().expect("Handle::new");
    h.sgemm_naive(&config, &a, &b, &mut c);

    let expected = K as f32;
    let bad = c.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();

    if bad == 0 {
        println!("SGEMM OK: {M}x{N}x{K} verified");
    } else {
        eprintln!("SGEMM FAIL: {bad} mismatches");
        std::process::exit(1);
    }
}
