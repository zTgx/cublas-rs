//! Linear regression via batch gradient descent — the classic L2 workload.
//!
//! Given a feature matrix `X (M × N)` and targets `y (M)`, fit weights `w (N)`
//! so that `X w ≈ y`. Each GD step is:
//!
//!     pred = X @ w               (sgemv NoTrans)
//!     residual = pred - y        (saxpy: residual = -1·y + pred + 0)
//!     loss = ||residual||²       (sdot of residual with itself)
//!     grad = Xᵀ @ residual / M   (sgemv Trans, then scale)
//!     w   -= lr · grad           (saxpy)
//!
//! Tail: small `strsv` + `ssymv` smoke checks.
//!
//! Run with:
//!   cargo oxide run --bin linreg
//!   RUST_LOG=cublas_rs=debug cargo oxide run --bin linreg

use std::time::Instant;

use cublas_rs::{DeviceBuf, Handle, Transpose, Triangular, Diag, prelude::*};
use tracing_subscriber::EnvFilter;

const M: usize = 4096;     // samples
const N: usize = 64;       // features
const EPOCHS: usize = 1000;
const LR: f32 = 0.1;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,cublas_rs=info,linreg=info")),
        )
        .init();

    let h = Handle::new()?;

    // ── Synthetic data: y = X w_true exactly (no noise → optimum is w_true)
    let w_true: Vec<f32> = (0..N).map(|i| 0.05 + (i as f32) * 0.005).collect();
    let mut x = vec![0.0f32; M * N];
    let mut y = vec![0.0f32; M];
    for i in 0..M {
        for j in 0..N {
            // Deterministic pseudo-random in [-1, 1].
            let v = (((i * 73 + j * 19) & 0xff) as f32 / 128.0) - 1.0;
            x[i * N + j] = v;
        }
        let mut row_dot = 0.0f32;
        for j in 0..N {
            row_dot += x[i * N + j] * w_true[j];
        }
        y[i] = row_dot;
    }

    // ── Upload everything once ------------------------------------------
    let x_dev = h.upload(&x)?;
    let y_dev = h.upload(&y)?;
    let w_init = vec![0.0f32; N];
    let mut w_dev = h.upload(&w_init)?;
    let mut pred_dev = DeviceBuf::<f32>::zeroed(h.stream(), M)?;
    let mut residual_dev = DeviceBuf::<f32>::zeroed(h.stream(), M)?;
    let mut grad_dev = DeviceBuf::<f32>::zeroed(h.stream(), N)?;

    tracing::info!(samples = M, features = N, epochs = EPOCHS, "starting GD");

    let initial_loss = compute_loss(&h, &x_dev, &y_dev, &w_dev, &mut pred_dev, &mut residual_dev)?;
    println!("Loss at epoch   0: {:.6e}", initial_loss);

    let start = Instant::now();
    for epoch in 1..=EPOCHS {
        // pred = X @ w
        h.sgemv_tiled(
            Transpose::NoTrans,
            M,
            N,
            1.0,
            &x_dev,
            &w_dev,
            0.0,
            &mut pred_dev,
        )?;

        // residual = pred - y. Implemented as:
        //   residual = copy(pred); residual += -1 · y
        h.scopy(M, &pred_dev, &mut residual_dev)?;
        h.saxpy(M, -1.0, &y_dev, &mut residual_dev)?;

        // grad = (2/M) · Xᵀ @ residual
        // (the 2 comes from d/dw of ||·||²)
        h.sgemv_tiled(
            Transpose::Trans,
            M,
            N,
            2.0 / M as f32,
            &x_dev,
            &residual_dev,
            0.0,
            &mut grad_dev,
        )?;

        // w -= lr · grad   (saxpy: w = -lr·grad + w)
        h.saxpy(N, -LR, &grad_dev, &mut w_dev)?;

        if epoch == 1 || epoch % 100 == 0 || epoch == EPOCHS {
            let loss =
                compute_loss(&h, &x_dev, &y_dev, &w_dev, &mut pred_dev, &mut residual_dev)?;
            println!("Loss at epoch {epoch:>3}: {loss:.6e}");
        }
    }
    h.synchronize()?;
    let elapsed = start.elapsed();
    println!();
    println!(
        "GD complete: {} epochs in {:.3} s ({:.3} ms/epoch)",
        EPOCHS,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / EPOCHS as f64
    );

    // Compare learned w to ground truth.
    let w_learned = h.download(&w_dev)?;
    let mut max_err = 0.0f32;
    for j in 0..N {
        let e = (w_learned[j] - w_true[j]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    println!("max |w_learned - w_true|: {max_err:.4e}");
    println!("w_learned[0..5] = {:?}", &w_learned[..5]);
    println!("w_true   [0..5] = {:?}", &w_true[..5]);

    // ── strsv smoke check ----------------------------------------------
    // Build an upper-triangular system U x = b where x = [1, 2, ..., n].
    const TN: usize = 8;
    let mut u = vec![0.0f32; TN * TN];
    for i in 0..TN {
        for j in i..TN {
            u[i * TN + j] = if i == j { 2.0 } else { 0.5 };
        }
    }
    let x_true_tri: Vec<f32> = (0..TN).map(|i| (i + 1) as f32).collect();
    let mut b = vec![0.0f32; TN];
    for i in 0..TN {
        for j in i..TN {
            b[i] += u[i * TN + j] * x_true_tri[j];
        }
    }
    h.strsv_simple(
        Triangular::Upper,
        Transpose::NoTrans,
        Diag::NonUnit,
        TN,
        &u,
        &mut b,
    )?;
    let strsv_ok = b
        .iter()
        .zip(x_true_tri.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3);
    println!("strsv (Upper, NoTrans) smoke: {}", if strsv_ok { "OK" } else { "FAIL" });

    // ── ssymv smoke check ---------------------------------------------
    // A = I, alpha=1, beta=0 → y = x.
    const SN: usize = 16;
    let mut a_sym = vec![0.0f32; SN * SN];
    for i in 0..SN {
        a_sym[i * SN + i] = 1.0;
    }
    let x_sym: Vec<f32> = (0..SN).map(|i| (i as f32) * 0.5).collect();
    let mut y_sym = vec![0.0f32; SN];
    h.ssymv_simple(Triangular::Lower, SN, 1.0, &a_sym, &x_sym, 0.0, &mut y_sym)?;
    let ssymv_ok = y_sym
        .iter()
        .zip(x_sym.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    println!("ssymv (Lower, A=I) smoke: {}", if ssymv_ok { "OK" } else { "FAIL" });

    if !strsv_ok || !ssymv_ok || max_err > 0.1 {
        std::process::exit(1);
    }
    Ok(())
}

/// loss = (1 / M) · ||X w - y||²
fn compute_loss(
    h: &Handle,
    x: &DeviceBuf<f32>,
    y: &DeviceBuf<f32>,
    w: &DeviceBuf<f32>,
    pred: &mut DeviceBuf<f32>,
    residual: &mut DeviceBuf<f32>,
) -> Result<f32> {
    h.sgemv_tiled(Transpose::NoTrans, M, N, 1.0, x, w, 0.0, pred)?;
    h.scopy(M, pred, residual)?;
    h.saxpy(M, -1.0, y, residual)?;
    let sumsq = h.sdot(M, residual, residual)?;
    Ok(sumsq / M as f32)
}
