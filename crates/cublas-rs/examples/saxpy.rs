//! SAXPY end-to-end smoke test.
//!
//! Run with:
//!   cargo oxide run --bin saxpy

use cublas_rs::saxpy;

fn main() {
    const N: usize = 1024;
    let alpha = 2.0f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mut y = vec![1.0f32; N];

    saxpy(N, alpha, &x, &mut y);

    let mut errors = 0;
    for i in 0..N {
        let expected = alpha * x[i] + 1.0;
        if (y[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!("[{i}] expected {expected}, got {}", y[i]);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SAXPY OK: {N} elements verified");
    } else {
        eprintln!("SAXPY FAIL: {errors} mismatches");
        std::process::exit(1);
    }
}
