# hello — GPU printf smoke test

Minimal end-to-end check of the cuda-oxide toolchain: codegen backend →
PTX → driver → kernel launch → device `gpu_printf!`. No BLAS dependency.
Works on every CUDA card from sm_20 onward.

Run this first whenever something looks broken — if `hello` fails, the
issue is in your toolchain (cargo-oxide, LLVM, CUDA driver), not in
cuBLAS-rs.

## Run

From the workspace root:

```bash
cargo oxide run --bin hello
```

Expected output:

```
[host] initializing CUDA...
[host] launching kernel...
Hello from GPU thread 0
Hello from GPU thread 1
...
Hello from GPU thread 7
[host] done — 8 GPU printf lines should appear above.
```

## What to look at

`src/main.rs` shows the bare minimum for a cuda-oxide kernel:

- A `#[cuda_module] mod kernels` block holding one `#[kernel] fn`.
- A `LaunchConfig` with grid/block dims and shared-mem bytes.
- `kernels::load(&ctx)?` — works because the kernel is defined in this
  bin's own crate (PTX gets embedded into the binary's `.oxart` section).

For kernels in dependency crates (everything in `cublas-l1` / `cublas-l3`),
see the `saxpy` and `sgemm_basic` examples — they load PTX from disk
because `kernels::load` doesn't reach into dep-crate `.oxart`.
