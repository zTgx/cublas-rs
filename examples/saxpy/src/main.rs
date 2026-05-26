//! SAXPY end-to-end smoke test.
//!
//! Mirrors the C cuBLAS calling pattern:
//!   cublasHandle_t h;  cublasCreate(&h);  cublasSaxpy(h, ...);
//!
//! Run with:
//!   cargo oxide run --bin saxpy

use cublas_rs::{Handle, Transpose};

fn main() {
    const N: usize = 1024;
    let alpha = 2.0f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mut y = vec![1.0f32; N];

    let h = Handle::new().expect("Handle::new — needs a CUDA device and `cublas_l1.ptx` in cwd");
    h.saxpy_simple(N, alpha, &x, &mut y).expect("saxpy");

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

    // Reduction smoke checks against the original x = [0, 1, ..., N-1].
    let dot = h.sdot_simple(N, &x, &x).expect("sdot");
    let nrm2 = h.snrm2_simple(N, &x).expect("snrm2");
    let asum = h.sasum_simple(N, &x).expect("sasum");
    let argmax = h.isamax_simple(N, &x).expect("isamax");

    let expected_sumsq: f32 = (0..N).map(|i| (i as f32).powi(2)).sum();
    let expected_nrm2 = expected_sumsq.sqrt();
    let expected_asum: f32 = (0..N).map(|i| i as f32).sum();

    let report = |name, got: f32, want: f32| {
        let rel = ((got - want) / want).abs();
        println!(
            "  {name}: got {got:.4e}  want {want:.4e}  rel_err {rel:.2e}{}",
            if rel < 1e-3 { "" } else { "  ⚠" }
        );
    };
    println!("Reductions:");
    report("sdot(x,x)", dot, expected_sumsq);
    report("snrm2(x)", nrm2, expected_nrm2);
    report("sasum(x)", asum, expected_asum);
    if argmax == N - 1 {
        println!("  isamax: {argmax} (correct — largest |x[i]| at index {})", N - 1);
    } else {
        eprintln!("  isamax: {argmax} (FAIL — expected {})", N - 1);
        std::process::exit(1);
    }

    // Elementwise smoke checks.
    let mut scal_buf: Vec<f32> = (0..N).map(|i| i as f32).collect();
    h.sscal_simple(N, 3.0, &mut scal_buf).expect("sscal");
    let scal_ok = scal_buf
        .iter()
        .enumerate()
        .all(|(i, &v)| (v - 3.0 * i as f32).abs() < 1e-3);

    let src: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
    let mut dst = vec![0.0f32; N];
    h.scopy_simple(N, &src, &mut dst).expect("scopy");
    let copy_ok = dst.iter().zip(src.iter()).all(|(a, b)| a == b);

    println!(
        "Elementwise: sscal {}  scopy {}",
        if scal_ok { "OK" } else { "FAIL" },
        if copy_ok { "OK" } else { "FAIL" }
    );
    if !scal_ok || !copy_ok {
        std::process::exit(1);
    }

    // L2 sgemv smoke check: A = ones(M, K), x = [1, 1, ..., 1], expected y[i] = K.
    // Then verify Trans: y = Aᵀ * x where x is ones(M) → y[j] = M.
    const M_ROWS: usize = 16;
    const K_COLS: usize = 24;
    let a_mat = vec![1.0f32; M_ROWS * K_COLS];
    let x_vec = vec![1.0f32; K_COLS];
    let mut y_vec = vec![0.0f32; M_ROWS];
    h.sgemv_simple(
        Transpose::NoTrans,
        M_ROWS,
        K_COLS,
        1.0,
        &a_mat,
        &x_vec,
        0.0,
        &mut y_vec,
    )
    .expect("sgemv NoTrans");
    let n_ok = y_vec.iter().all(|&v| (v - K_COLS as f32).abs() < 1e-3);

    let x_trans = vec![1.0f32; M_ROWS];
    let mut y_trans = vec![0.0f32; K_COLS];
    h.sgemv_simple(
        Transpose::Trans,
        M_ROWS,
        K_COLS,
        1.0,
        &a_mat,
        &x_trans,
        0.0,
        &mut y_trans,
    )
    .expect("sgemv Trans");
    let t_ok = y_trans.iter().all(|&v| (v - M_ROWS as f32).abs() < 1e-3);

    println!(
        "L2 sgemv: NoTrans {}  Trans {}",
        if n_ok { "OK" } else { "FAIL" },
        if t_ok { "OK" } else { "FAIL" }
    );
    if !n_ok || !t_ok {
        std::process::exit(1);
    }
}
