# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.
Targets NVIDIA Ampere (sm_80 / sm_86).

## What's included

- **Level 1** (`cublas-l1`) — `saxpy`, `scal`, `copy`, `axpy`, `dot`, `nrm2`, `asum`, `iamax`
- **Level 2** (`cublas-l2`) — `gemv`, `trsv` (plus `symv` planned)
- **Level 3** (`cublas-l3`) — `sgemm` / `dgemm` / `hgemm` with progressive variants
  (naive, tiled shared-memory, vectorized, double-buffered), plus batched and
  strided-batched GEMM
- **Bench / validation** — `GpuTimer` + CPU reference checker in `benches/`

See [`CLAUDE.md`](./CLAUDE.md) for architecture, build instructions, and the
implementation status table.

## Quick start

Modelled after the C cuBLAS API — build a `Handle` once, then call BLAS ops
as methods:

```rust
use cublas_rs::Handle;

fn main() {
    let h = Handle::new().expect("Handle::new");
    let alpha = 2.0f32;
    let x = vec![1.0f32; 1024];
    let mut y = vec![1.0f32; 1024];

    h.saxpy(x.len(), alpha, &x, &mut y);
    // y[i] = 2.0 * 1.0 + 1.0 = 3.0
}
```

`Handle::new` loads the PTX files cargo-oxide drops in the workspace root
(`cublas_l1.ptx`, ...), so run binaries from the workspace root.

## Build

Kernel crates require the `cargo-oxide` subcommand (custom rustc backend
that compiles `#[kernel]` Rust to PTX):

```bash
# One-time install of the cargo-oxide subcommand
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

# Build + run the toolchain hello-world (works on any CUDA card sm_20+)
cargo oxide run --bin hello   # pure toolchain check, no BLAS dep
cargo oxide run --bin saxpy   # L1 smoke test
cargo oxide run --bin sgemm   # L3 SGEMM (naive)
```

Each example under `examples/` is a standalone crate (its own `Cargo.toml`
+ `README.md`), wired into the workspace so `--bin <name>` finds them. See
`examples/<name>/README.md` for per-example notes.

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
