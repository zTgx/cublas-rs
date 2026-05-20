# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.

## What's included

- **SGEMM/DGEMM** - Matrix multiplication with progressive optimization: naive, tiled, double-buffered, vectorized
- **HGEMM** - Half-precision (f16) with scalar and Tensor Core (WMMA/MMA) variants
- **Batched GEMM** - Concurrent multi-batch execution via CUDA streams
- **Vector ops** - SAXPY, DOT, NRM2

## Quick start

```rust
use cublas_rs::{prelude::*, sgemm_naive};

fn main() {
    let config = GemmConfig {
        m: 512,
        n: 512,
        k: 512,
        alpha: 1.0,
        beta: 0.0,
    };

    let mut a = vec![1.0f32; config.m * config.k];
    let mut b = vec![1.0f32; config.k * config.n];
    let mut c = vec![0.0f32; config.m * config.n];

    sgemm_naive(&config, &a, &b, &mut c);
}
```

## Requirements

- Nightly Rust (see `rust-toolchain.toml`)
- CUDA toolkit

## License

Apache-2.0
