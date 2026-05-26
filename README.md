# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.
Targets NVIDIA Ampere (sm_80 / sm_86).

## What's included

- **Level 1** (`cublas-l1`) — `saxpy`, `scal`, `copy`, `axpy`, `dot`, `nrm2`, `asum`, `iamax`
- **Level 2** (`cublas-l2`) — `gemv`, `trsv` (plus `symv` planned)
- **Level 3** (`cublas-l3`) — `sgemm` / `dgemm` / `hgemm` with progressive variants
  (naive, tiled shared-memory, vectorized, double-buffered), plus batched and
  strided-batched GEMM
- **Bench / validation** — `GpuTimer` + CPU reference checker in `cublas-bench-core`

See [`CLAUDE.md`](./CLAUDE.md) for architecture, build instructions, and the
implementation status table.

## Quick start

```rust
use cublas_rs::saxpy;

fn main() {
    let alpha = 2.0f32;
    let x = vec![1.0f32; 1024];
    let mut y = vec![1.0f32; 1024];

    saxpy(x.len(), alpha, &x, &mut y);
    // y[i] = 2.0 * 1.0 + 1.0 = 3.0
}
```

By BLAS-level convention:

```rust
use cublas_rs::level1::saxpy;
use cublas_rs::level2::sgemv;
use cublas_rs::level3::sgemm_naive;
```

## Build

Kernel crates require the `cargo-oxide` subcommand (custom rustc backend
that compiles `#[kernel]` Rust to PTX):

```bash
# One-time install of the cargo-oxide subcommand
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

# Build + run the SAXPY smoke test
cargo oxide build
cargo oxide run                     # default-run picks saxpy
cargo oxide run --bin sgemm_basic   # the other one
```

Examples live at `crates/cublas-rs/examples/` and are declared as `[[bin]]`
targets — cargo-oxide forwards `--bin`, not `--example`. `default-members`
in the workspace `Cargo.toml` lets these commands work from the repo root
without `-p cublas-rs`.

For pure type-checking (host crates + IDE), plain cargo works:

```bash
cargo check --workspace
```

## Requirements

- Nightly Rust (`rust-toolchain.toml` pins `nightly-2026-04-03`)
- CUDA Toolkit 12.x+
- LLVM 21+, Clang 21
- Ampere or newer GPU

## License

Apache-2.0
