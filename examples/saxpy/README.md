# saxpy — L1 smoke test

Computes `y := alpha * x + y` for a length-1024 vector and verifies the
result against a CPU reference. The smallest end-to-end exercise of the
cuBLAS-rs `Handle` API.

## Run

From the workspace root:

```bash
cargo oxide run --bin saxpy
```

Expected output:

```
SAXPY OK: 1024 elements verified
```

## What to look at

`src/main.rs` shows the canonical cuBLAS-rs calling pattern, matching the
C `cublasSaxpy` shape:

```rust
let h = cublas_rs::Handle::new()?;
h.saxpy(n, alpha, &x, &mut y);
```

`Handle::new()` does three things behind the scenes:

1. Initialises a CUDA context on device 0.
2. Creates a default stream.
3. Loads `cublas_l1.ptx` (dropped at the workspace root by cargo-oxide)
   and types each L1 kernel module.

That's why you must run from the workspace root: `Handle::new()` uses a
cwd-relative path to find the PTX file.
