# cuBLAS-rs — Project Guide

A Rust BLAS implementation on top of [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
Pedagogical / research-grade. Target hardware: **NVIDIA Ampere (sm_80 / sm_86)**.

## Project shape

```
cublas-rs/
├── crates/
│   ├── cublas-core/          Host-only: GemmConfig, BlasScalar, MatrixLayout, Transpose
│   ├── cublas-bench-core/    GpuTimer + CPU reference validator
│   ├── cublas-l1/            BLAS Level 1 — saxpy, scal, copy, axpy, dot, nrm2, asum, iamax
│   ├── cublas-l2/            BLAS Level 2 — gemv, trsv, symv
│   ├── cublas-l3/            BLAS Level 3 — sgemm/dgemm/hgemm/batched (organised internally
│   │   src/                  by precision family with implementation variants)
│   │     ├── sgemm/{naive,tiled,vectorized,double_buf}.rs
│   │     ├── dgemm/{naive,tiled,vectorized,double_buf}.rs
│   │     ├── hgemm/{half,tensor_core}.rs
│   │     └── batched/{simple,strided}.rs
│   └── cublas-rs/            Top-level facade — flat API + level1/level2/level3 namespaces
├── benches/                  cublas-bench-core based benches
└── ../cuda-oxide/            (sibling) The codegen + runtime we depend on

Examples live inside the facade crate at `crates/cublas-rs/examples/` and are
declared as `[[bin]]` targets (`autoexamples = false`). This is because
cargo-oxide's standalone mode only forwards `--bin <name>` to the underlying
`cargo run`, not `--example`. Run them with:

```bash
cargo oxide run                     # default-run = hello (pure toolchain check)
cargo oxide run --bin saxpy         # L1 kernel exercise
cargo oxide run --bin sgemm_basic   # L3 kernel exercise (will panic — stub)
```

`hello` uses `gpu_printf!` and works on any CUDA card sm_20+. Use it to
confirm cuda-oxide → PTX → driver → launch is healthy before debugging any
BLAS-side issue.

`default-members = ["crates/cublas-rs"]` in the workspace `Cargo.toml` makes
these commands work from the repo root without `-p cublas-rs`.
```

Crate naming follows the BLAS levels (`l1`/`l2`/`l3`). `cublas-core` keeps its
conventional Rust name and holds shared types only. `cublas-rs` is the public
export point — most callers should only need to depend on it.

## Build & run

### Toolchain prerequisites

- Nightly pinned in `rust-toolchain.toml` — matches cuda-oxide (`nightly-2026-04-03`).
- CUDA Toolkit 12.x+.
- LLVM 21+ (`llc-21` on PATH).
- Clang 21 (`libclang-common-21-dev`).
- `cargo-oxide`:
  ```bash
  cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
  ```

Run `cargo oxide doctor` to validate.

### What builds with what

| Crate                | Tool             | Reason                                   |
|----------------------|------------------|------------------------------------------|
| `cublas-core`        | `cargo build`    | Pure host types, no CUDA                 |
| `cublas-bench-core`  | `cargo build`    | Host-side timing wrappers                |
| `cublas-l1`          | `cargo oxide`    | Contains `#[cuda_module]` kernels        |
| `cublas-l2`          | `cargo oxide`    | Contains `#[cuda_module]` kernels        |
| `cublas-l3`          | `cargo oxide`    | Contains `#[cuda_module]` kernels        |
| `cublas-rs` (facade) | `cargo oxide`    | Pulls in kernel crates transitively      |
| `examples/*` (bins)  | `cargo oxide run --bin <name>` | Single-source kernels      |

`cargo check --workspace` works end-to-end (the `#[cuda_module]` proc macro
expands to host code that just *references* embedded PTX bytes; the actual
PTX gen only happens under `cargo oxide build`). This means rust-analyzer and
type-checking work in any normal IDE.

Single kernel crate:
```bash
cargo oxide build -p cublas-l1
```

## Hardware capability matrix

Targeting Ampere (sm_80/86). What is and is not in reach with current
cuda-oxide primitives:

| Primitive                           | cuda-oxide wrapper | Ampere supports |
|-------------------------------------|--------------------|-----------------|
| `SharedArray` + `sync_threads`      | ✓                  | ✓ — use it      |
| `thread::*`, `warp::*` intrinsics   | ✓                  | ✓               |
| Vectorized loads (`f32x4` etc.)     | via LLVM           | ✓               |
| `cp.async` (non-bulk, Ampere)       | **✗ missing**      | ✓ — wrapper TBD |
| WMMA / `mma.sync.aligned`           | **✗ missing**      | ✓ — wrapper TBD |
| `cp.async.bulk.tensor` (TMA)        | ✓                  | ✗ — Hopper+     |
| WGMMA                               | ✓                  | ✗ — Hopper      |
| tcgen05 / TMEM                      | ✓                  | ✗ — Blackwell   |
| Cluster / DSMEM / CLC               | ✓                  | ✗ — Hopper+     |

**Practical ceiling on A100 without wrapper extensions:** ~60–70% of cuBLAS
SGEMM. Tensor Core HGEMM (312 TFLOPS path) blocked until WMMA wrapper exists.

## Kernel author template

Single-source. Host wrapper + device kernel in the same file:

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn my_kernel(alpha: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            *out = alpha * x[i] + *out;
        }
    }
}

pub fn my_op(alpha: f32, x: &[f32], y: &mut [f32]) {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let x_d = DeviceBuffer::from_host(&stream, x).unwrap();
    let mut y_d = DeviceBuffer::from_host(&stream, y).unwrap();

    let module = kernels::load(&ctx).expect("load PTX");
    module
        .my_kernel(&stream, LaunchConfig::for_num_elems(x.len() as u32),
                   alpha, &x_d, &mut y_d)
        .expect("launch");

    let out = y_d.to_host_vec(&stream).unwrap();
    y.copy_from_slice(&out);
}
```

Working reference: `crates/cublas-l1/src/saxpy.rs`. For tiled / shared-memory
kernels see `../cuda-oxide/crates/rustc-codegen-cuda/examples/tiled_gemm/`.

## API conventions

**v1 (current):** Functions take host slices. Internally allocate device
buffers, launch, copy back. Easy to call, wasteful for repeated calls.

**v2 (planned):** A `Handle` wraps `CudaContext` + `CudaStream`. Kernel
functions take `&Handle` + `&DeviceBuffer<T>`. Lets callers amortize context
creation and chain ops on the same stream. The v1 functions stay as
`*_simple` convenience wrappers.

When the v2 split happens, the v1 signatures already in this repo stay as-is
— add a parallel module rather than retrofit.

## Implementation status

| Level | Op                              | File                                          | Status |
|-------|---------------------------------|-----------------------------------------------|--------|
| L1    | `saxpy`                         | `cublas-l1/src/saxpy.rs`                      | ✓      |
| L1    | `scal/copy/axpy/dot/nrm2/asum/iamax` | `cublas-l1/src/*.rs`                     | stub   |
| L2    | `sgemv/dgemv/hgemv`             | `cublas-l2/src/gemv.rs`                       | stub   |
| L2    | `strsv`                         | `cublas-l2/src/trsv.rs`                       | stub   |
| L2    | `ssymv`                         | `cublas-l2/src/symv.rs`                       | stub   |
| L3    | `sgemm_naive`                   | `cublas-l3/src/sgemm/naive.rs`                | stub   |
| L3    | `sgemm_tiled`                   | `cublas-l3/src/sgemm/tiled.rs`                | stub   |
| L3    | `sgemm_vectorized`              | `cublas-l3/src/sgemm/vectorized.rs`           | stub   |
| L3    | `sgemm_double_buf`              | `cublas-l3/src/sgemm/double_buf.rs`           | stub   |
| L3    | `dgemm_*` (4 variants)          | `cublas-l3/src/dgemm/*.rs`                    | stub   |
| L3    | `hgemm_half`                    | `cublas-l3/src/hgemm/half.rs`                 | stub   |
| L3    | `hgemm_tensor_core`             | `cublas-l3/src/hgemm/tensor_core.rs`          | blocked — WMMA wrapper missing |
| L3    | `batched_sgemm`                 | `cublas-l3/src/batched/simple.rs`             | stub   |
| L3    | `strided_batched_sgemm`         | `cublas-l3/src/batched/strided.rs`            | stub   |
| —     | `GpuTimer`                      | `cublas-bench-core/src/timer.rs`              | stub   |
| —     | `validate_gemm`                 | `cublas-bench-core/src/validator.rs`          | stub   |

## Reference paths in `../cuda-oxide`

| You want to learn... | Look at |
|---|---|
| First kernel end-to-end | `crates/rustc-codegen-cuda/examples/vecadd/` |
| Naive GEMM             | `examples/gemm/` (219 LoC) |
| Tiled / shared memory  | `examples/tiled_gemm/` (281 LoC) |
| Production GEMM        | `examples/gemm_sol/` (6522 LoC, Hopper/Blackwell) |
| Async / streams        | `examples/async_mlp/`, `examples/async_vecadd/` |
| Cross-crate kernels    | `examples/cross_crate_kernel/` |

## Style rules

- One `#[cuda_module]` block per file, named `kernels`.
- Kernel function inside the module = same op name as the public host wrapper
  (no `_kernel` suffix — the namespace disambiguates).
- Host wrapper does buffer alloc + launch + copy back. No business logic on
  host beyond shape assertions.
- Stubbed functions: keep the signature, `let _ = (...)` to silence warnings,
  `todo!("...")` body — never leave a function without a body.
- `expect(...)` with a short context string for v1; bubble `Result` once we
  add a `CublasError` type.

## Open design questions

1. **Error type.** `expect` everywhere now; v1 should add `CublasError`
   (probably `thiserror`-based) before any production use.
2. **`Handle` shape.** cuBLAS's `cublasHandle_t` carries stream, math mode,
   pointer mode, workspace. Mirror or simplify?
3. **Mixed precision.** `cublasGemmEx` allows distinct input/compute/output
   types — extend `GemmConfig` or add a new type?
4. **WMMA wrapper.** Either contribute upstream to cuda-oxide, or carry a
   local `dialect-nvvm` patch. Pre-req for Tensor Core HGEMM on Ampere.
