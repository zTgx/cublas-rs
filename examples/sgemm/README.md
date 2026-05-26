# sgemm — L3 SGEMM smoke test

Computes `C := A * B` for 512×512×512 matrices using the naive SGEMM
variant (one thread per output element, no shared memory) and spot-checks
the result against a CPU reference. The smallest end-to-end exercise of
the cuBLAS-rs L3 path.

## Run

From the workspace root:

```bash
cargo oxide run --bin sgemm
```

Expected output:

```
SGEMM OK: 512x512x512 verified
```

## What to look at

`src/main.rs` uses the same `Handle` pattern as the `saxpy` example, just
with an L3 method:

```rust
let h = cublas_rs::Handle::new()?;
h.sgemm_naive(&config, &a, &b, &mut c);
```

`GemmConfig { m, n, k, alpha, beta }` packages the standard GEMM scalars.
Matrices are row-major host slices.

`naive` is the baseline implementation — correctness-focused, not
performance-focused. The same `Handle` exposes `sgemm_tiled`,
`sgemm_vectorized`, and `sgemm_double_buf` for progressively faster
variants (still stubs at time of writing).
