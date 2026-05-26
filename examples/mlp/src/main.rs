//! 2-layer MLP forward-pass inference loop — the realistic cuBLAS calling
//! pattern: build one `Handle`, upload all weights ONCE to GPU memory, then
//! run thousands of forwards that only touch the device buffers.
//!
//! Hot path uses `Handle::sgemm_tiled` (shared-memory tiled) + `Handle::saxpy`
//! for the bias-broadcast trick. ReLU / second-layer bias / softmax are
//! flagged `// TODO` until the matching ops land.
//!
//! Run with:
//!   cargo oxide run --bin mlp                          # info-level logs
//!   RUST_LOG=cublas_rs=debug cargo oxide run --bin mlp # per-op debug spans
//!   RUST_LOG=trace             cargo oxide run --bin mlp # full H2D/D2H trace

use std::time::Instant;

use cublas_rs::{DeviceBuf, Handle, prelude::*};
use tracing_subscriber::EnvFilter;

const BATCH: usize = 128;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;

const WARMUP_ITERS: usize = 5;
const TIMED_ITERS: usize = 50;

fn main() -> Result<()> {
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

    // One handle for the entire program — exactly like cublasCreate.
    let h = Handle::new()?;

    // Pretend these came off disk. They never get re-uploaded inside the
    // forward loop — that's the whole point of device-resident weights.
    let w1 = init_weights(IN_FEATURES, HIDDEN, 0.01);
    let b1 = init_bias(HIDDEN, 0.1);
    let w2 = init_weights(HIDDEN, OUT_FEATURES, 0.01);
    let x = init_input(BATCH, IN_FEATURES);
    let b1_broadcast = broadcast_bias(&b1, BATCH);

    // ── One-shot uploads. After this, the GPU owns everything. ────────────
    let w1_dev = h.upload(&w1)?;
    let w2_dev = h.upload(&w2)?;
    let b1_broadcast_dev = h.upload(&b1_broadcast)?;
    let x_dev = h.upload(&x)?;
    // Scratch buffers — zero-initialised on the device, never round-trip.
    let mut z1_dev = DeviceBuf::<f32>::zeroed(h.stream(), BATCH * HIDDEN)?;
    let mut z2_dev = DeviceBuf::<f32>::zeroed(h.stream(), BATCH * OUT_FEATURES)?;

    tracing::info!(iters = WARMUP_ITERS, "warmup");
    for _ in 0..WARMUP_ITERS {
        forward(
            &h,
            &x_dev,
            &w1_dev,
            &b1_broadcast_dev,
            &w2_dev,
            &mut z1_dev,
            &mut z2_dev,
        )?;
    }
    // Drain the warmup launches before starting the clock — otherwise their
    // queued work bleeds into the timed window.
    h.synchronize()?;

    tracing::info!(iters = TIMED_ITERS, "timed run");
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        forward(
            &h,
            &x_dev,
            &w1_dev,
            &b1_broadcast_dev,
            &w2_dev,
            &mut z1_dev,
            &mut z2_dev,
        )?;
    }
    // Kernel launches are async on the host. Without this sync the timer
    // would only measure launch overhead, not actual GPU work.
    h.synchronize()?;
    let elapsed = start.elapsed();

    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / TIMED_ITERS as f64;
    let samples_per_sec = (BATCH * TIMED_ITERS) as f64 / elapsed.as_secs_f64();
    let flops_per_iter =
        2.0 * (BATCH * HIDDEN * IN_FEATURES) as f64 + 2.0 * (BATCH * OUT_FEATURES * HIDDEN) as f64;
    let gflops = flops_per_iter / (ms_per_iter / 1000.0) / 1e9;

    // Download Z2 once for spot-check.
    let z2_host = h.download(&z2_dev)?;

    println!();
    println!("MLP forward pass — tiled SGEMM backend, device-resident weights");
    println!("  shape:       batch={BATCH}, {IN_FEATURES} → {HIDDEN} → {OUT_FEATURES}");
    println!("  per-iter:    {ms_per_iter:>7.3} ms");
    println!("  throughput:  {samples_per_sec:>7.0} samples/s");
    println!("  GFLOPS:      {gflops:>7.1}");
    println!();
    println!("Spot-check: z2[0..5] = {:?}", &z2_host[..5]);

    Ok(())
}

/// One forward pass — pure device work, no H2D/D2H per iteration.
#[tracing::instrument(level = "debug", skip_all, name = "forward")]
fn forward(
    h: &Handle,
    x: &DeviceBuf<f32>,
    w1: &DeviceBuf<f32>,
    b1_broadcast: &DeviceBuf<f32>,
    w2: &DeviceBuf<f32>,
    z1: &mut DeviceBuf<f32>,
    z2: &mut DeviceBuf<f32>,
) -> Result<()> {
    // Layer 1: Z1 = X @ W1
    h.sgemm_tiled(
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
    )?;

    // Z1 += b1 (broadcast across rows, materialised once at startup).
    h.saxpy(BATCH * HIDDEN, 1.0, b1_broadcast, z1)?;

    // TODO: in-place ReLU on z1 — elementwise kernel.

    // Layer 2: Z2 = Z1 @ W2
    h.sgemm_tiled(
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
    )?;

    // TODO: Z2 += b2 (same bias-broadcast saxpy pattern).
    // TODO: softmax(z2) along OUT_FEATURES axis — row reduction kernel.

    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────

fn init_weights(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
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

fn broadcast_bias(bias: &[f32], rows: usize) -> Vec<f32> {
    let cols = bias.len();
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        out.extend_from_slice(bias);
    }
    out
}
