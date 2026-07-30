# Milestone 1: Tensor ops

**Goal:** the non-matmul operations a transformer needs, each with a reference
implementation and a test.

## Concepts

Numerical stability (the softmax max-subtraction trick), fused elementwise
kernels, reductions, and why these ops are bandwidth-bound rather than
compute-bound.

## Tasks

- [x] **`softmax`** (row-wise): subtract the row max before `exp`: without it
      fp32 overflows to `inf` above ~88 and you get NaNs. Reference + SIMD.
- [x] **`rmsnorm`**: `x / sqrt(mean(x²) + eps) * weight`. Note the reduction
      then the scale; two passes, or one with a fused accumulator.
- [x] **`silu` / `swiglu`**: `silu(z) = z * sigmoid(z)`;
      `swiglu(x) = silu(x·W1) * (x·W3)`.
- [x] **`rope`**: rotate pairs of dimensions in Q and K by a
      position-dependent angle. Applied to Q and K only, **never V**.
- [x] **`add` / residual**, and a `matvec` path (matrix × vector): decode's
      shape, and bandwidth-bound, so optimise it differently from GEMM.
- [x] Extend the correctness harness: each op vs its reference, at sizes that
      exercise vector-width edges (7, 8, 9, 33, 64).
- [x] Add a numerical-stability test: softmax of large logits (e.g. 1000) must
      not produce NaN.

## Files

`include/ops.hpp`, `src/ops.cpp`, `src/main.cpp` (extend the harness),
`docs/06-transformer.md`.

## Definition of Done

- [x] Every op matches its reference within a stated tolerance at all test sizes.
- [x] `softmax` of large values is finite and sums to 1.
- [x] `rmsnorm`, `swiglu`, `rope` verified against hand-computed values on a
      tiny case (a fixture checked into the repo).
- [x] `make all` warning-free; `make test` green.

## Notes

These ops are memory-bound: they do O(n) work per O(n) bytes. Do not expect
matmul-style speedups from SIMD here: the wins come from **fusing** (e.g.
normalise-and-scale in one pass) to avoid extra memory round-trips.

**Next:** [Milestone 2: Transformer](milestone-2-transformer.md).
