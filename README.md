# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.

## What's included

- **SGEMM/DGEMM** - Matrix multiplication with progressive optimization: naive, tiled, double-buffered, vectorized
- **HGEMM** - Half-precision (f16) with scalar and Tensor Core (WMMA/MMA) variants
- **Batched GEMM** - Concurrent multi-batch execution via CUDA streams
- **Vector ops** - SAXPY, DOT, NRM2

## Workspace structure

```
crates/
  cublas-core/          Shared config and traits
  cublas-sgemm/         SGEMM kernels
  cublas-dgemm/         DGEMM kernels
  cublas-hgemm/         HGEMM kernels
  cublas-batched-gemm/  Batched GEMM kernels
  cublas-vector/        Vector operation kernels
  cublas-rs/            Unified API (re-exports all kernels)
  cublas-bench-core/    Benchmark utilities
benches/                Performance benchmarks vs NVIDIA cuBLAS
examples/              Usage examples
```

## Requirements

- Nightly Rust (see `rust-toolchain.toml`)
- CUDA toolkit

## License

Apache-2.0
