# mlp — realistic cuBLAS usage pattern (zero-background walkthrough)

Written for someone who has never touched neural networks or cuBLAS. By
the end you should understand:

- **What is being computed** (a minimal neural network running inference)
- **Why the code is shaped the way it is** (the standard cuBLAS calling
  pattern that real production code uses)
- **What every number and log line means**
- **What is genuinely real here and what is teaching-only scaffolding** —
  this matters; see [§5](#5-whats-real-vs-whats-scaffolding) before
  copying any of this into production

If you just want to run it, jump to [§3](#3-running-it).

---

## 1. Background concepts

### 1.1 What is a neural-network "forward pass"

Think of a neural network as a function `f(x)`: you feed it an input `x`
(e.g. a picture), it outputs a prediction `y` (e.g. "this is a 7"). The
chain of arithmetic that turns `x` into `y` is called the **forward pass**
(also called **inference**).

The simplest neural network is the **MLP** (Multi-Layer Perceptron). It's
a stack of `matrix-multiply → bias-add → nonlinear activation` steps.

This example uses a 2-layer MLP:

```
input X        :  128 samples, each a 784-D vector            (128, 784)
                                                              │
   ┌──────────────────────────────────────────────────────────┤
   │  Layer 1: project 784 → 256                              │
   │  Z1 = X @ W1                                              ← matmul   W1: (784, 256)
   │  Z1 = Z1 + b1 (same bias vector added to every row)       ← bias      b1: (256,)
   │  Z1 = ReLU(Z1)  ← nonlinear activation                   (TODO — not yet implemented)
   └──────────────────────────────────────────────────────────┤
                                                              │
   intermediate shape                                         : (128, 256)
                                                              │
   ┌──────────────────────────────────────────────────────────┤
   │  Layer 2: project 256 → 10                               │
   │  Z2 = Z1 @ W2                                             ← matmul   W2: (256, 10)
   │  Z2 = Z2 + b2                                            (TODO)
   │  Z2 = softmax(Z2)  ← turn scores into probabilities      (TODO)
   └──────────────────────────────────────────────────────────┤
                                                              │
final Z2       : per-sample scores over 10 classes            (128, 10)
```

The numbers `128 / 784 / 256 / 10` aren't arbitrary:

| Meaning                          | Value         | Why this value                                         |
|----------------------------------|---------------|--------------------------------------------------------|
| `BATCH` — samples per call       | 128           | Big enough to saturate GPU parallelism                 |
| `IN_FEATURES` — input dim        | 784 = 28×28   | Flattened MNIST digit image (classic textbook size)    |
| `HIDDEN` — middle-layer width    | 256           | Heuristic — enough capacity, not wasteful              |
| `OUT_FEATURES` — output dim      | 10            | Ten classes (digits 0–9)                               |

### 1.2 Why this has anything to do with cuBLAS

The two **matrix multiplies** (`X @ W1` and `Z1 @ W2`) are by far the
heaviest computation in the forward pass.

Matrix multiplication in numerical-library land is called **GEMM**
(GEneral Matrix-Matrix multiply), and it's the headline operation of
**BLAS** (Basic Linear Algebra Subprograms) — a standard API that's been
around since the 1970s. NVIDIA's GPU implementation of BLAS is called
**cuBLAS** and ships as a C library.

`cublas-rs` is a Rust port of that idea. This `mlp` example shows the
**standard way real code (inference servers, training loops) calls
cuBLAS-rs**.

### 1.3 What "Handle" means

cuBLAS's C API always starts with:

```c
cublasHandle_t h;
cublasCreate(&h);          // grab a handle
... many cublasSgemm/cublasSaxpy calls ...
cublasDestroy(h);
```

`Handle` is a **long-lived context object** holding the GPU context, a
CUDA stream, loaded kernels, etc. **Analogy:** it's like a database
`Connection` — you don't `connect/disconnect` for every SQL query; you
open one connection and run ten thousand queries on it.

cuBLAS-rs follows the same pattern:

```rust
let h = Handle::new()?;          // once
h.sgemm_naive(...);              // reused N times
h.saxpy(...);                    // reused N times
```

**That's the core pattern this example demonstrates**: build a handle
once, reuse it across 50 iterations.

---

## 2. What the code does

`src/main.rs` has six steps, matching the doc-comment at the top of the
file:

```
1. Initialise tracing (the logging system)
2. Handle::new()                    ← GPU init + PTX load, ONCE
3. Set up weights / input / scratch ← pretend these were loaded from disk
4. Warmup: run 5 forwards           ← warm up GPU, JIT-compile PTX, fill caches
5. Timed run: 50 forwards           ← measure steady-state performance
6. Print results
```

The `forward` function body is what the diagram above looks like in code:

```rust
h.sgemm_naive(&GemmConfig { m, n, k, alpha, beta }, x, w1, z1);  // Z1 = X @ W1
h.saxpy(z1.len(), 1.0, b1_broadcast, z1);                        // Z1 += b1
h.sgemm_naive(&GemmConfig { ... }, z1, w2, z2);                  // Z2 = Z1 @ W2
// TODO: bias / activation / softmax
```

### 2.1 Why "warmup" matters

The first time you launch a GPU kernel, several things happen:

1. The CUDA driver JIT-compiles PTX into machine code for your specific
   GPU architecture.
2. Caches are cold.
3. Lazily initialised resources fire up.

So the first few iterations are inevitably slower than steady state. To
measure a kernel's real speed you discard the warmup runs, then start
the timer. This is the standard practice for GPU benchmarking.

### 2.2 Why bias is "broadcast"

Mathematically `Z1 + b1` means "add vector `b1` to every row of `Z1`".
But `saxpy` only does `y = α·x + y` — same-length elementwise add, no
broadcasting.

So we manually tile `b1` (length 256) 128 times into a (128, 256) matrix
`b1_broadcast` and apply one saxpy. This is a common pre-fusion-kernel
cuBLAS pattern. (Production code would use a fused `bias_add` kernel
instead, but plain saxpy works.)

### 2.3 Row-major layout

`X (128, 784)` lives in memory as a flat `Vec<f32>` of length `128 * 784`,
laid out **row by row**: all 784 elements of row 0 first, then row 1,
and so on. The offset of `X[i, j]` is `i * 784 + j`.

cuBLAS-rs kernels are written with this convention. You don't need to
transpose anything.

---

## 3. Running it

### 3.1 Prerequisites

Same as the rest of the repo:

- `cargo-oxide` installed (see the root README for the install command)
- CUDA Toolkit 12.x+
- LLVM 21, Clang 21
- A supported NVIDIA GPU

`cargo-oxide` is a custom cargo subcommand. It uses cuda-oxide's rustc
backend to compile Rust kernels into PTX. Plain `cargo run` will not
work — it will panic at runtime because no PTX was generated.

### 3.2 Basic invocation

**Must be run from the workspace root** (the repo root, not from inside
`examples/mlp/`):

```bash
cargo oxide run --bin mlp
```

Why the workspace root? Because `Handle::new()` reads `cublas_l1.ptx`
and `cublas_l3.ptx` from the current directory. cargo-oxide drops those
files at the workspace root during the build, so cwd has to match.

### 3.3 Expected output

The first run takes a while to compile (dependencies + cuda-oxide
codegen — anywhere from 30 seconds to a minute). After that you'll see:

```
2026-05-26T03:58:37  INFO mlp: starting MLP inference benchmark batch=128 in_features=784 hidden=256 out_features=10
2026-05-26T03:58:37  INFO Handle::with_device{device_idx=0}: cublas_rs: CUDA context + default stream ready device_idx=0
2026-05-26T03:58:37  INFO Handle::with_device{device_idx=0}: cublas_rs: Handle ready (L1 + L3 modules loaded)
2026-05-26T03:58:37  INFO mlp: warmup iters=5
2026-05-26T03:58:37  INFO mlp: timed run iters=50

MLP forward pass — naive SGEMM backend
  shape:       batch=128, 784 → 256 → 10
  per-iter:      9.069 ms
  throughput:    14115 samples/s
  GFLOPS:          5.7

Spot-check: z2[0..5] = [-0.0052861464, -0.005165218, -0.00076814665, -0.0014223573, 0.0001731799]
```

### 3.4 Reading those numbers

| Number                          | Meaning                                                      |
|---------------------------------|--------------------------------------------------------------|
| `per-iter: 9.069 ms`           | One forward pass takes ~9 ms on average                       |
| `throughput: 14115 samples/s`  | ~14k samples processed per second (= 128 / 9.069 ms)          |
| `GFLOPS: 5.7`                  | About 5.7 billion floating-point ops per second               |
| `z2[0..5]`                     | First five output values — eyeball check that nothing exploded |

**Where does the GFLOPS number come from?** One forward pass does roughly
`2 * BATCH * HIDDEN * IN_FEATURES + 2 * BATCH * OUT_FEATURES * HIDDEN`
floating-point multiply-adds (each `sgemm` is `2 * M * N * K` because
each output element requires one multiply and one add per inner index).
Divide by elapsed seconds and by 1e9.

**Is 5.7 GFLOPS good?** Not even close to what cuBLAS achieves —
production cuBLAS SGEMM on an A100 hits ~19 TFLOPS (19,000 GFLOPS). The
implementation we ship is `sgemm_naive`: one thread per output element,
no shared memory. It's a deliberate teaching baseline. Once
`sgemm_tiled` / `sgemm_vectorized` / `sgemm_double_buf` are filled in,
this number will climb a lot.

---

## 4. Using tracing logs

Every host-side function in the library is instrumented with the
`tracing` crate. Control verbosity with the `RUST_LOG` environment
variable.

### 4.1 Default (no RUST_LOG set)

Only `info`-level events show up — handle init, warmup/timed phase
markers. That's what §3.3 shows.

### 4.2 See every op call

```bash
RUST_LOG=cublas_rs=debug cargo oxide run --bin mlp
```

Every `h.saxpy(...)` and `h.sgemm_naive(...)` opens and closes a span:

```
DEBUG saxpy{op="saxpy"}: cublas_l1::saxpy: new
DEBUG saxpy{op="saxpy"}: cublas_l1::saxpy: close time.busy=... time.idle=...
DEBUG sgemm_naive{op="sgemm_naive" m=128 n=256 k=784}: cublas_l3::sgemm::naive: new
...
```

Useful for "which op is slow?".

### 4.3 See the full data flow

```bash
RUST_LOG=trace cargo oxide run --bin mlp
```

This adds H2D (host→device copies), kernel launch parameters (grid /
block dims), and D2H (device→host copies) for every op. One
`sgemm_naive` call looks roughly like:

```
TRACE: H2D A, B, C
TRACE: launch SGEMM naive grid=(16, 8, 1) block=(16, 16, 1)
TRACE: D2H C
```

Noisy — but invaluable the first time you stand up a new kernel.

### 4.4 Filter by component

`RUST_LOG` uses the `EnvFilter` syntax — you can filter per target:

```bash
RUST_LOG=cublas_l3=trace cargo oxide run --bin mlp                # L3 only
RUST_LOG=cublas_l1=debug,cublas_l3=info cargo oxide run --bin mlp
```

---

## 5. What's real vs what's scaffolding

This is the most important section if you're thinking of cargo-culting
this code. **The calling *shape* is realistic; the *substance* is a
teaching baseline.** Concretely:

### 5.1 Real (safe to copy the pattern)

- `Handle::new()` once, reused across N calls — identical to cuBLAS /
  PyTorch / TensorRT.
- Warmup separated from timed runs — standard benchmark hygiene.
- Multiple ops chained on the same handle and stream
  (`sgemm → saxpy → sgemm`).
- Structured `tracing` for observability — what production services do.
- Row-major dense storage end-to-end — matches cuBLAS row-major mode.

### 5.2 Scaffolding (do **not** ship as-is)

| Limitation                                                | Why it's a problem                                          | What real code does                                          |
|-----------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|
| **Weights are re-uploaded H2D every forward**             | `Handle::sgemm_naive(..., a, b, c)` takes host slices, so the function internally does `DeviceBuffer::from_host(stream, a)` each call. Most of the 9 ms is copy, not compute. | Keep weights in `DeviceBuffer<T>` for the lifetime of the model; only the input batch gets uploaded per call. |
| **No activation function (ReLU)**                         | Without nonlinearity the whole network collapses into one big linear matmul — has zero expressive power compared to a real MLP. | A small custom elementwise kernel (it's not in classic BLAS). |
| **No softmax** on the output                              | Final scores aren't a probability distribution.             | A row-wise reduction kernel (also not in BLAS).               |
| **Bias broadcast via pre-tiled buffer + saxpy**           | Wastes memory (we materialise a 128×256 copy of a 256-vector) and the saxpy is unnecessarily large. | A fused `bias_add` kernel that streams the bias from a single 256-vector. |
| **`sgemm_naive` is ~3000× slower than cuBLAS**            | One thread per output element, no shared memory, no tiling. | `sgemm_tiled` / `_vectorized` / `_double_buf` (still stubs in this repo). |
| **Single CUDA stream, no async pipelining**              | Compute and copies serialise on the same stream.            | Multiple streams + event-based H2D/compute overlap.          |
| **Weights are deterministic pseudo-random**               | Not a trained model. Outputs are noise.                     | Load real trained weights from disk (safetensors / ONNX / ...). |

### 5.3 What it would take to make this production-ready

In rough priority order:

1. **Device-resident buffers.** Add `&DeviceBuffer<T>` overloads on
   `Handle` (already on the v2 roadmap in `CLAUDE.md`). Weights upload
   once, stay on the GPU.
2. **Elementwise ReLU kernel** + a **row-reduction softmax kernel**.
3. **Fused bias-add** kernel (broadcast version), to drop the pre-tile
   step.
4. **Faster SGEMM variants** — `sgemm_tiled` first, then vectorize, then
   double-buffer.

Until those exist, this example is a faithful *blueprint* for the
calling shape and observability — not a runnable inference engine.

---

## 6. Glossary

| Term                | Meaning                                                              |
|---------------------|----------------------------------------------------------------------|
| **BLAS**            | Basic Linear Algebra Subprograms — standard linear-algebra API (1970s) |
| **cuBLAS**          | NVIDIA's BLAS implementation on GPU (C library)                       |
| **GEMM**            | GEneral Matrix-Matrix multiply (`C = α·A·B + β·C`), BLAS level 3      |
| **SGEMM**           | Single-precision (FP32) GEMM                                         |
| **SAXPY**           | Single-precision `y = α·x + y`, BLAS level 1                         |
| **MLP**             | Multi-Layer Perceptron — the simplest feed-forward neural network    |
| **Forward pass**    | Feeding input through the network to get output (= inference)        |
| **Handle**          | cuBLAS-style long-lived context object (think DB connection)         |
| **CUDA context**    | A process's execution environment on a GPU                           |
| **CUDA stream**     | An in-order command queue on the GPU                                 |
| **Kernel**          | A parallel function that runs on the GPU                             |
| **PTX**             | NVIDIA's intermediate assembly (a "virtual bytecode" for GPUs)        |
| **JIT**             | Just-In-Time compilation, PTX → architecture-specific machine code   |
| **H2D / D2H**       | Host-to-Device / Device-to-Host memory copy (CPU ↔ GPU)             |
| **Grid / Block**    | CUDA kernel launch hierarchy (a grid of blocks, each a grid of threads) |
| **Row-major**       | Matrix stored row by row; `A[i,j]` offset = `i * cols + j`           |
| **GFLOPS**          | Billions of floating-point operations per second                     |
| **Tracing**         | Rust's structured-logging crate (spans + structured fields)          |
| **EnvFilter**       | The `RUST_LOG` parser used by `tracing-subscriber`                   |

---

## 7. Things you can try

A few zero-background-friendly ways to play with this:

1. **Change the network size.** Bump `HIDDEN` from 256 to 1024 — wider
   hidden layers give the GPU more work, so the GFLOPS number should go
   up.
2. **Change `BATCH`.** Try 1 (latency-optimised) and 1024 (throughput-
   optimised) — observe how per-iter and throughput trade off.
3. **Add a third layer.** Copy one of the existing sgemm calls and
   route `Z2 → Z3`.
4. **See the cost of skipping warmup.** Set `WARMUP_ITERS = 0` — the
   per-iter number should creep up because the first iteration's JIT
   compile pollutes the timed window.
5. **Turn tracing off.** Delete the `tracing_subscriber::fmt()...init()`
   block — the program still runs, just without logs. `tracing::info!`
   is a no-op when there's no subscriber registered.

Once a new op lands (`sscal`, ReLU, softmax, ...), uncomment the
matching `// TODO` and watch the output change.
