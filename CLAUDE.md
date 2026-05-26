# cuBLAS-rs — Project Guide

A Rust BLAS implementation on top of [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
Pedagogical / research-grade. Target hardware: **NVIDIA Ampere (sm_80 / sm_86)**.

## Project shape

```
cublas-rs/
├── crates/
│   ├── cublas-core/          Host-only: GemmConfig, BlasScalar, MatrixLayout, Transpose, CublasError
│   ├── cublas-l1/            BLAS Level 1 — saxpy, scal, copy, axpy, dot, nrm2, asum, iamax
│   ├── cublas-l2/            BLAS Level 2 — gemv, trsv, symv
│   ├── cublas-l3/            BLAS Level 3 — one flat file per precision family
│   │   src/                    sgemm.rs / dgemm.rs / hgemm.rs / batched.rs
│   │                          (matches the L1 / L2 layout)
│   └── cublas-rs/            Top-level facade — `Handle` API surface
├── examples/                 Standalone example crates
│   ├── hello/                Pure cuda-oxide smoke test, no BLAS dep
│   ├── saxpy/                L1 end-to-end + reduction smoke + L2 sgemv smoke
│   ├── sgemm/                L3 perf compare: naive vs tiled, f32 + f64
│   ├── mlp/                  Realistic L1 + L3 inference loop (device-resident weights)
│   └── linreg/               L2-heavy GD demo + strsv/ssymv smoke
├── benches/                  Workspace member, self-contained: bench harnesses + GpuTimer + validators
└── ../cuda-oxide/            (sibling) The codegen + runtime we depend on

Each example is a workspace member with its own `Cargo.toml`, `src/main.rs`,
and `README.md` — mirrors how a downstream user would consume the library.
Declared as bins (not `[[example]]`) because cargo-oxide's standalone mode
only forwards `--bin <name>` to the underlying `cargo run`. Run from the
workspace root:

```bash
cargo oxide run --bin hello   # toolchain check (works on any CUDA card sm_20+)
cargo oxide run --bin saxpy   # L1 kernel exercise
cargo oxide run --bin sgemm   # L3 SGEMM (naive)
```

`hello` uses `gpu_printf!` and has no BLAS dependency. Use it to confirm
cuda-oxide → PTX → driver → launch is healthy before debugging any
BLAS-side issue.

The workspace has no `default-members` set on purpose — `cargo run --bin X`
must search across all members to find bins in the example crates.
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
| `benches`            | `cargo build`    | Bench harness + GpuTimer + validators    |
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

Single-source. Host op + device kernel in the same file. The host op takes
a typed kernel module and a stream from the caller — never calls
`kernels::load` itself, because that only works when the kernel and the bin
are in the same crate (it reads the bin's `.oxart` section).

```rust
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {              // pub: facade needs to call `from_module`
    use super::*;

    #[kernel]
    pub fn my_op(alpha: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            *out = alpha * x[i] + *out;
        }
    }
}

pub fn my_op(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    alpha: f32,
    x: &[f32],
    y: &mut [f32],
) {
    let x_d = DeviceBuffer::from_host(stream, x).expect("copy x");
    let mut y_d = DeviceBuffer::from_host(stream, y).expect("copy y");

    module
        .my_op(stream, LaunchConfig::for_num_elems(x.len() as u32),
               alpha, &x_d, &mut y_d)
        .expect("launch");

    let out = y_d.to_host_vec(stream).expect("copy y back");
    y.copy_from_slice(&out);
}
```

Then in the level crate's `lib.rs`, add a `Modules` struct that loads all
kernels for the level from one PTX file and types each view:

```rust
pub struct Modules {
    pub my_op: my_op::kernels::LoadedModule,
    // ...
}

impl Modules {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let raw = ctx.load_module_from_file("cublas_l1.ptx")?;
        Ok(Self { my_op: my_op::kernels::from_module(raw)? })
    }
}
```

And in `cublas-rs/src/lib.rs`, expose a friendly method on `Handle`:

```rust
impl Handle {
    pub fn my_op(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        cublas_l1::my_op(&self.l1.my_op, &self.stream, alpha, x, y);
    }
}
```

Working reference: `crates/cublas-l1/src/saxpy.rs` (file-level kernel),
`crates/cublas-l1/src/lib.rs` (`Modules`), `crates/cublas-rs/src/lib.rs`
(`Handle::saxpy`). For tiled / shared-memory kernels see
`../cuda-oxide/crates/rustc-codegen-cuda/examples/tiled_gemm/`.

## API conventions

**Why this shape:** cargo-oxide's standalone mode only embeds PTX into the
`.oxart` section of the *entry crate*. Kernels defined in a dep crate aren't
embedded into the final binary, so `kernels::load(&ctx)` from inside a dep
crate's host op fails with `ModuleNotFound`. The workaround is to load the
per-crate PTX file (`cublas_l1.ptx`, ...) from cwd via
`ctx.load_module_from_file` and type it via `kernels::from_module`. Binaries
that use `Handle` must run from the workspace root (where cargo-oxide drops
the PTX files).

**v1 (current):**
- `Handle` on `cublas-rs` is the **sole** public entry point. Every L1 / L2
  / L3 op (including future batched/strided) is a method on `Handle`. The
  facade does **not** re-export `cublas_l1::*` / `cublas_l2::*` etc., so
  callers can't bypass the handle.
- Level crates (`cublas-l1`, ...) host free fns + per-level `Modules` are
  internal wiring. When a kernel exists, the free fn takes
  `(module, stream, ...)` and the Handle method forwards into it. When the
  kernel is still a stub, the Handle method calls `todo!()` directly and the
  free fn isn't used.
- Host fns take `&[T]` host slices and allocate device buffers per call.
  Wasteful for repeated calls; fine for the smoke-test use case.

**v2 (planned):** Public ops on `Handle` will gain `&DeviceBuffer<T>`
variants so callers can amortize H2D/D2H. Slice-taking variants stay as the
shorthand.

**Adding a new kernel — full checklist:**
1. Fill in `#[cuda_module] pub mod kernels` in the op's `.rs` (e.g.
   `cublas-l1/src/sscal.rs`).
2. Give the file's host fn the `(module, stream, ...)` signature.
3. Add a field on the level crate's `Modules` struct and call
   `op::kernels::from_module(raw_module.clone())` in `Modules::load`.
4. Replace the `todo!()` in the matching `Handle::<op>` method body with a
   call into the level crate's free fn.

## Implementation status

| Level | Op                              | File                                          | Status |
|-------|---------------------------------|-----------------------------------------------|--------|
| L1    | `saxpy`                         | `cublas-l1/src/saxpy.rs`                      | ✓      |
| L1    | `sscal`                         | `cublas-l1/src/scal.rs`                       | ✓      |
| L1    | `scopy`                         | `cublas-l1/src/copy.rs`                       | ✓      |
| L1    | `sdot`                          | `cublas-l1/src/dot.rs`                        | ✓      |
| L1    | `snrm2`                         | `cublas-l1/src/nrm2.rs`                       | ✓      |
| L1    | `sasum`                         | `cublas-l1/src/asum.rs`                       | ✓      |
| L1    | `isamax`                        | `cublas-l1/src/iamax.rs`                      | ✓      |
| L1    | `daxpy/haxpy`                   | `cublas-l1/src/axpy.rs`                       | stub   |
| L2    | `sgemv` (naive + tiled)         | `cublas-l2/src/gemv.rs`                       | ✓      |
| L2    | `dgemv`                         | `cublas-l2/src/gemv.rs`                       | ✓      |
| L2    | `hgemv`                         | `cublas-l2/src/gemv.rs`                       | ✓ — raw-u16 in/out, f32 accumulate, IEEE-754 bit-twiddle (subnormals flushed) |
| L2    | `strsv` (4 variants, seq)       | `cublas-l2/src/trsv.rs`                       | ✓      |
| L2    | `ssymv` (Upper/Lower)           | `cublas-l2/src/symv.rs`                       | ✓      |
| L3    | `sgemm_*` (naive/tiled/vectorized/double_buf) | `cublas-l3/src/sgemm.rs`        | ✓      |
| L3    | `dgemm_naive` + `dgemm_tiled`   | `cublas-l3/src/dgemm.rs`                      | ✓      |
| L3    | `dgemm_vectorized/double_buf`   | `cublas-l3/src/dgemm.rs`                      | stub   |
| L3    | `hgemm_half`                    | `cublas-l3/src/hgemm.rs`                      | stub   |
| L3    | `hgemm_tensor_core`             | `cublas-l3/src/hgemm.rs`                      | blocked — WMMA wrapper missing |
| L3    | `strided_batched_sgemm`         | `cublas-l3/src/batched.rs`                    | ✓ — 3D grid, blockIdx.z = batch, 16×16 tile per block |
| L3    | `batched_sgemm`                 | `cublas-l3/src/batched.rs`                    | ✓ — host concat + delegate to strided         |
| —     | `GpuTimer`                      | `benches/src/timer.rs`                        | stub   |
| —     | `validate_gemm`                 | `benches/src/validator.rs`                    | stub   |

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
