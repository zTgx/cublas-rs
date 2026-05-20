# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.

## What's included

- **SGEMM/DGEMM** - Matrix multiplication with progressive optimization: naive, tiled, double-buffered, vectorized
- **HGEMM** - Half-precision (f16) with scalar and Tensor Core (WMMA/MMA) variants
- **Batched GEMM** - Concurrent multi-batch execution via CUDA streams
- **Vector ops** - SAXPY, DOT, NRM2

## Quick start

```rust
use cublas_rs::{Gemm, GemmConfig};
use cuda_core::{DeviceBuffer, Stream};

// SGEMM: C = alpha * A * B + beta * C
fn main() {
    let m = 512;
    let n = 512;
    let k = 512;

    let mut a = DeviceBuffer::<f32>::alloc(m * k);
    let mut b = DeviceBuffer::<f32>::alloc(k * n);
    let mut c = DeviceBuffer::<f32>::alloc(m * n);

    // ... fill a, b ...

    let stream = Stream::create();

    Gemm::sgemm(
        &stream,
        GemmConfig { m, n, k, alpha: 1.0, beta: 0.0 },
        &a,   // A: m x k
        &b,   // B: k x n
        &mut c, // C: m x n
    );

    stream.synchronize();
}
```

## Requirements

- Nightly Rust (see `rust-toolchain.toml`)
- CUDA toolkit

## License

Apache-2.0
