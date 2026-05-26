//! 2-layer MLP forward-pass inference loop — mirrors how production code
//! actually drives cuBLAS:
//!
//!   1. Build one `Handle` at startup (= `cublasCreate`), reuse it forever.
//!   2. Pre-allocate input / weight / output host buffers.
//!   3. Warmup (PTX JIT + caches).
//!   4. Run N forward passes in a tight loop, time the steady-state phase.
//!   5. Report ms/iter + samples/s.
//!
//! The hot path is two `sgemm_naive` calls (X @ W1 → Z1, Z1 @ W2 → Z2) plus
//! a `saxpy` for the bias term on the first layer. The activation /
//! second-layer-bias / softmax steps are flagged `todo!()` style — drop
//! them in once `cublas-l1::sscal`-family ops and elementwise kernels land.
//!
//! Run with:
//!   cargo oxide run --bin mlp                          # info-level logs
//!   RUST_LOG=cublas_rs=debug cargo oxide run --bin mlp # per-op debug spans
//!   RUST_LOG=trace             cargo oxide run --bin mlp # full H2D/D2H trace

use std::time::Instant;

use cublas_rs::{Handle, prelude::*};
use tracing_subscriber::EnvFilter;

// Network shape — small enough that a CPU can sanity-check, large enough
// that the GPU work dominates host-side overhead.
const BATCH: usize = 128;
const IN_FEATURES: usize = 784; // e.g. flattened MNIST
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;

const WARMUP_ITERS: usize = 5;
const TIMED_ITERS: usize = 50;

fn main() {
    // Standard `tracing-subscriber` setup: honour `RUST_LOG`, default to
    // info-level for cublas_rs + this bin.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,cublas_rs=info,mlp=info")),
        )
        .init();

    tracing::info!(
        batch = BATCH,
        in_features = IN_FEATURES,
        hidden = HIDDEN,
        out_features = OUT_FEATURES,
        "starting MLP inference benchmark"
    );

    // ── Step 1: handle (= `cublasHandle_t`). Build once, share everywhere.
    let h = Handle::new().expect("Handle::new — CUDA + PTX files reachable from cwd?");

    // ── Step 2: model weights (pretend we loaded these from disk).
    let w1 = init_weights(IN_FEATURES, HIDDEN, 0.01);
    let b1 = init_bias(HIDDEN, 0.1);
    let w2 = init_weights(HIDDEN, OUT_FEATURES, 0.01);

    // ── Step 3: input batch + scratch buffers.
    let x = init_input(BATCH, IN_FEATURES);
    let mut z1 = vec![0.0f32; BATCH * HIDDEN];
    let mut z2 = vec![0.0f32; BATCH * OUT_FEATURES];
    // Pre-broadcast b1 so we can apply it with one saxpy on Z1. (In real
    // code you'd use a fused bias-add kernel; broadcasting here keeps the
    // example self-contained.)
    let b1_broadcast = broadcast_bias(&b1, BATCH);

    // ── Step 4: warmup.
    tracing::info!(iters = WARMUP_ITERS, "warmup");
    for _ in 0..WARMUP_ITERS {
        forward(&h, &x, &w1, &b1_broadcast, &w2, &mut z1, &mut z2);
    }

    // ── Step 5: timed loop.
    tracing::info!(iters = TIMED_ITERS, "timed run");
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        forward(&h, &x, &w1, &b1_broadcast, &w2, &mut z1, &mut z2);
    }
    let elapsed = start.elapsed();

    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / TIMED_ITERS as f64;
    let samples_per_sec = (BATCH * TIMED_ITERS) as f64 / elapsed.as_secs_f64();
    // FLOPs per forward pass: 2 sgemms.
    let flops_per_iter =
        2.0 * (BATCH * HIDDEN * IN_FEATURES) as f64 + 2.0 * (BATCH * OUT_FEATURES * HIDDEN) as f64;
    let gflops = flops_per_iter / (ms_per_iter / 1000.0) / 1e9;

    println!();
    println!("MLP forward pass — naive SGEMM backend");
    println!("  shape:       batch={BATCH}, {IN_FEATURES} → {HIDDEN} → {OUT_FEATURES}");
    println!("  per-iter:    {ms_per_iter:>7.3} ms");
    println!("  throughput:  {samples_per_sec:>7.0} samples/s");
    println!("  GFLOPS:      {gflops:>7.1}");
    println!();
    println!("Spot-check: z2[0..5] = {:?}", &z2[..5]);
}

/// One forward pass. Two sgemms + one saxpy; the rest of a real MLP
/// (second-layer bias, ReLU, softmax) is stubbed below.
#[tracing::instrument(level = "debug", skip_all, name = "forward")]
fn forward(
    h: &Handle,
    x: &[f32],
    w1: &[f32],
    b1_broadcast: &[f32],
    w2: &[f32],
    z1: &mut [f32],
    z2: &mut [f32],
) {
    // Layer 1: Z1 = X @ W1
    h.sgemm_naive(
        &GemmConfig {
            m: BATCH,
            n: HIDDEN,
            k: IN_FEATURES,
            alpha: 1.0,
            beta: 0.0,
        },
        x,
        w1,
        z1,
    );

    // Z1 += b1 (broadcast). One full-size saxpy beats N row-wise saxpys.
    h.saxpy(z1.len(), 1.0, b1_broadcast, z1);

    // TODO: in-place ReLU on z1 — needs an elementwise kernel
    //       (`Handle::relu(&mut z1)` once that lands).

    // Layer 2: Z2 = Z1 @ W2
    h.sgemm_naive(
        &GemmConfig {
            m: BATCH,
            n: OUT_FEATURES,
            k: HIDDEN,
            alpha: 1.0,
            beta: 0.0,
        },
        z1,
        w2,
        z2,
    );

    // TODO: Z2 += b2 — same bias-broadcast saxpy pattern as Layer 1.
    // TODO: softmax(z2) along the OUT_FEATURES axis — needs a reduction
    //       kernel; not classic BLAS.
}

// ── helpers ──────────────────────────────────────────────────────────────

fn init_weights(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    // Deterministic pseudo-random init — same idea as PyTorch's default
    // Linear weight init, without pulling in a real RNG dep.
    (0..rows * cols)
        .map(|i| (((i * 2654435761) & 0xffff) as f32 / 65535.0 - 0.5) * 2.0 * scale)
        .collect()
}

fn init_bias(len: usize, value: f32) -> Vec<f32> {
    vec![value; len]
}

fn init_input(batch: usize, features: usize) -> Vec<f32> {
    (0..batch * features)
        .map(|i| ((i as f32) % 32.0) / 32.0)
        .collect()
}

/// Tile `bias` (length cols) across `rows` rows so the result has shape
/// (rows, cols) row-major.
fn broadcast_bias(bias: &[f32], rows: usize) -> Vec<f32> {
    let cols = bias.len();
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        out.extend_from_slice(bias);
    }
    out
}
