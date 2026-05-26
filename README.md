# cuBLAS-rs

A BLAS implementation built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide), written in Rust.
Target architecture: NVIDIA Ampere (sm_80 / sm_86); develops and runs fine
on older cards (Pascal sm_61 tested), some perf-only kernels skipped on
pre-Volta.

## What's included

- **Level 1** (`cublas-l1`) — `saxpy`, `daxpy`, `haxpy`, `sscal`, `scopy`,
  `sdot`, `snrm2`, `sasum`, `isamax` all implemented.
- **Level 2** (`cublas-l2`) — `sgemv` (naive + tiled), `dgemv`, `hgemv`,
  `strsv` (4 variants), `ssymv` (Upper/Lower) all implemented.
- **Level 3** (`cublas-l3`) — `sgemm` with four variants (naive / tiled /
  vectorized / double-buffered), `dgemm` (naive + tiled), `hgemm_half`
  (tiled), batched + strided-batched SGEMM. `hgemm_tensor_core` blocked
  on WMMA support in cuda-oxide.
- **Handle-based API** modelled after C cuBLAS: build once, reuse across
  thousands of calls. Each op exists in two flavours — `xxx` for device
  buffers (production path), `xxx_simple` for host slices (one-shot).
- **`tracing` instrumentation** throughout — `RUST_LOG=cublas_rs=debug`
  to see per-op spans, `RUST_LOG=trace` for H2D/launch/D2H detail.

See [`CLAUDE.md`](./CLAUDE.md) for the architecture, build matrix, and
full implementation status table.

## Quick start

C-cuBLAS-style: build a `Handle` once, then call BLAS ops as methods.
The `_simple` family takes host slices (allocates + uploads + downloads
per call); switch to the unsuffixed (device-buffer) form when you have
data already resident on the GPU.

```rust
use cublas_rs::Handle;

fn main() -> cublas_rs::Result<()> {
    let h = Handle::new()?;
    let alpha = 2.0f32;
    let x = vec![1.0f32; 1024];
    let mut y = vec![1.0f32; 1024];

    h.saxpy_simple(x.len(), alpha, &x, &mut y)?;
    // y[i] = 2.0 * 1.0 + 1.0 = 3.0
    Ok(())
}
```

For the production pattern (weights resident, hot inference loop), see
[`examples/mlp/`](./examples/mlp/) — it uploads weights once via
`h.upload(...)` and runs `Handle::sgemm_tiled` + `Handle::saxpy` in a
timed loop.

`Handle::new` loads the PTX files cargo-oxide drops in the workspace
root (`cublas_l1.ptx`, `cublas_l2.ptx`, `cublas_l3.ptx`), so binaries
that use `Handle` must be run from the workspace root.

## Examples

Each example under `examples/` is a standalone crate (its own
`Cargo.toml` + `README.md`).

| Example      | What it shows                                                       |
|--------------|---------------------------------------------------------------------|
| `hello`      | Toolchain check — `#[cuda_module]` + `gpu_printf!`, no BLAS dep     |
| `saxpy`      | L1 end-to-end + reduction smoke + L2 sgemv / hgemv smoke            |
| `sgemm`      | L3 perf compare: SGEMM naive/tiled/vectorized/double_buf + DGEMM + batched |
| `mlp`        | Realistic inference loop: device-resident weights, tiled SGEMM       |
| `linreg`     | L2-heavy GD demo + strsv / ssymv smoke                              |

## Build

Kernel crates require the `cargo-oxide` subcommand (custom rustc backend
that compiles `#[kernel]` Rust to PTX):

```bash
# One-time install of the cargo-oxide subcommand
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

# Run from the workspace root.
cargo oxide run --bin hello    # toolchain smoke
cargo oxide run --bin saxpy    # L1 + L2 smoke
cargo oxide run --bin sgemm    # L3 perf compare
cargo oxide run --bin mlp      # realistic inference loop
cargo oxide run --bin linreg   # GD via L2 sgemv + L1 ops
```

For pure type-checking (host crates + IDE), plain cargo works:

```bash
cargo check --workspace
```

## Requirements

- Nightly Rust (`rust-toolchain.toml` pins `nightly-2026-04-03`)
- CUDA Toolkit 12.x+
- LLVM 21+, Clang 21
- NVIDIA GPU. Most kernels work from Kepler (sm_30) onward; tensor-core
  paths (HGEMM TC, WMMA) require sm_70+ and are not yet implemented.

## License

Apache-2.0
