# linreg — L2 linear regression via gradient descent

Fits weights `w` to a synthetic linear dataset `y = X w` using batch
gradient descent. End-to-end demo of the L2 ops on `Handle`, with small
`strsv` and `ssymv` smoke checks tacked on the end.

## Algorithm

Each GD step:

```
pred     = X @ w                  ← sgemv NoTrans
residual = pred - y               ← scopy + saxpy
loss     = (1/M) · ||residual||²  ← sdot
grad     = (2/M) · Xᵀ @ residual  ← sgemv Trans
w       -= LR · grad              ← saxpy
```

That's 4 distinct ops per step — exactly the L2 toolkit that BLAS gives
you for classical numerical workflows.

## Shape

| Constant | Value | Meaning                       |
|----------|-------|-------------------------------|
| `M`      | 4096  | training samples              |
| `N`      | 64    | features per sample           |
| `EPOCHS` | 1000  | GD steps                      |
| `LR`     | 0.1   | learning rate                 |

The synthetic `w_true` is `[0.05, 0.055, ..., 0.365]` and `y = X w_true`
with no noise — the optimum is exact, so convergence is a clean test
that the ops are correct.

## Run

```bash
cargo oxide run --bin linreg
```

Sample output:

```
Loss at epoch    0: 2.158821e-1
Loss at epoch    1: 8.411613e-2
Loss at epoch  100: 2.731443e-2
Loss at epoch  500: 1.763951e-3
Loss at epoch 1000: 1.409461e-4

GD complete: 1000 epochs in 0.339 s (0.339 ms/epoch)
max |w_learned - w_true|: 2.4337e-2
w_learned[0..5] = [0.048..., 0.053..., 0.058..., 0.063..., 0.068...]
w_true   [0..5] = [0.05,    0.055,    0.060,    0.065,    0.070]
strsv (Upper, NoTrans) smoke: OK
ssymv (Lower, A=I) smoke: OK
```

Loss drops ~1500× across 1000 epochs; the learned weights match the
ground-truth to ~3×10⁻². Per-epoch cost is ~0.3 ms on Pascal.

## What it exercises

| Op           | Where                              | Why this is realistic                          |
|--------------|------------------------------------|------------------------------------------------|
| `sgemv` (NoTrans) | forward `X @ w`               | Standard linear-model forward                  |
| `sgemv` (Trans)   | gradient `Xᵀ @ residual`      | Same op, transposed access pattern             |
| `sgemv_tiled`     | both of the above             | Shared-memory `x` tile, faster than naive      |
| `scopy`           | `residual = pred`             | Workspace setup before in-place subtract       |
| `saxpy`           | `residual -= y`, `w -= LR·g`  | Two distinct uses of the same op               |
| `sdot`            | `loss = residual · residual`  | L2-norm squared via inner product              |
| `strsv`           | smoke test                    | Verifies upper-triangular solve `U x = b`      |
| `ssymv`           | smoke test                    | Verifies symmetric matrix-vector with A = I    |

## What's still simplified

- **Batch GD, not stochastic.** Full pass through M = 4096 samples every
  step. In practice you'd do SGD with mini-batches.
- **Constant LR.** No schedule or momentum (would just be more saxpys
  with extra state).
- **No regularization.** Add an L2 term to the loss and you get
  `w -= LR · (grad + λ·w)` — one more saxpy.
- **Single precision throughout.** A real numerical pipeline might use
  f64 for the accumulators; `dgemv` is the gate, currently a stub.

## Log levels

| `RUST_LOG`                       | What you see                              |
|----------------------------------|-------------------------------------------|
| (unset)                          | `info`: handle init, epoch loss markers   |
| `cublas_rs=debug`                | each op entry/exit per epoch              |
| `trace`                          | per-op H2D/launch/D2H details (very noisy)|
