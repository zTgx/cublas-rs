# gemm_bench — cross-GPU GEMM perf comparison

Runs all SGEMM and DGEMM variants at multiple sizes, prints steady-state
GFLOPS. Designed to be portable so you can copy the repo to another GPU
and compare numbers directly.

## What it measures

- **SGEMM** (f32, 4 variants): `naive`, `tiled`, `vectorized`,
  `double_buf` at `{256, 512, 1024}` square sizes.
- **DGEMM** (f64, same 4 variants, same sizes).

## Methodology

For each (variant, size) pair:

1. Upload `A` (all 1.0) and `B` (all 1.0) to the GPU once.
2. Run `WARMUP_ITERS = 3` iterations to warm up the JIT compiler, fill
   caches, and reach steady state.
3. `synchronize()` to drain the warmup work.
4. Start the timer.
5. Launch the kernel `TIMED_ITERS = 10` times back-to-back.
6. `synchronize()` to make sure all launches finished.
7. Stop the timer.
8. Average ms/iter, compute GFLOPS as `2·n³ / time`.
9. Download `C` to host and verify every element equals `K` (since
   `A = B = 1.0` ⇒ `C[i,j] = K`).

The sync calls in steps 3 and 6 are critical — kernel launches are async
on the host, so without them you'd be measuring launch overhead only.

## Run

From the workspace root:

```bash
cargo oxide run --bin gemm_bench
```

Sample output (Pascal sm_61):

```
GEMM benchmark — cuBLAS-rs
  warmup=3, timed=10, A = B = 1.0 (C[i] should == K)

                size      naive      tiled     vector     dblbuf
  ────────────────────────────────────────────────────────────
  SGEMM        256       XX.X       XX.X       XX.X       XX.X
  SGEMM        512       XX.X       XX.X       XX.X       XX.X
  SGEMM       1024       XX.X       XX.X       XX.X       XX.X

DGEMM (Pascal cuts FP64 to ~1/32 SP; Ampere A100 ~1/2):
                size      naive      tiled     vector     dblbuf
  ────────────────────────────────────────────────────────────
  DGEMM        256        X.X        X.X        X.X        X.X
  ...
```

## What to compare across GPUs

- **Absolute GFLOPS** at the largest size — gives peak achievable
  throughput per variant on each GPU.
- **`tiled` / `naive` ratio** — should be ~2× on Pascal, larger on
  newer GPUs with better L1 caches.
- **`vector` vs `tiled`** — the thread-coarsening variant is the only
  one whose perf depends heavily on register pressure and ILP.
  On Pascal it's typically slower than `tiled`; on Ampere/Hopper it
  should pull ahead.
- **`double_buf` vs `tiled`** — the prefetch overlap is real only when
  the GPU has the bandwidth to overlap loads with compute. Marginal on
  Pascal, more on Ampere+.
- **DGEMM `tiled` vs `naive`** — Pascal-class consumer GPUs (e.g. GTX
  10-series) tank FP64 to ~1/32 SP, so tiling doesn't help (memory
  isn't the bottleneck, the FP64 ALUs are). On A100/H100 DGEMM tiled
  beats naive by 2-3×.

## Adding sizes

Edit `SIZES` at the top of `src/main.rs`. Watch out for memory: at size
`1024`, each DGEMM matrix is `1024² × 8 B = 8 MB`; three of them per
size is 24 MB. Going up to `4096` would be 384 MB — fine on most
GPUs but worth noting.
