# 00 — Overview

## What nutllm is

An LLM **inference engine**: the software that takes trained model weights and a
prompt and produces tokens. Not a training framework, not a wrapper around
someone else's runtime — the kernels, the transformer math, the tokenizer, the
cache, and the quantisation are all written here in C++17 with no dependencies.

## The one-sentence idea

> Make a language model run fast on ordinary hardware, starting from the
> arithmetic.

## Design goals

1. **Correctness is a gate, not a goal.** Every optimised kernel is checked
   against a plain reference before it is timed. Fast and wrong is worthless.
2. **Measure everything.** No claim without the number and the conditions that
   produced it. The README's table is output, not marketing.
3. **No dependencies.** No BLAS, no Eigen, no PyTorch. Writing the kernels is
   the project ([ADR 0001](decisions/0001-no-dependencies.md)).
4. **Start where the time goes.** Milestone 0 is matmul, because that is where
   inference actually spends its cycles — not because it is the easiest place to
   start.

## What it is *not*

- Not a training framework (no autograd, no optimiser).
- Not a serving stack (no batching scheduler or HTTP API in v1).
- Not a general tensor library — only the ops a transformer needs.

## The stack

```
prompt ──▶ tokenizer (BPE)                                   M2
              │ token ids
              ▼
       transformer blocks                                    M2
        ├─ RMSNorm, RoPE, attention (Q·Kᵀ, softmax, ·V)      M1/M2
        ├─ SwiGLU feed-forward                                M1
        └─ KV cache for incremental decoding                  M4
              │  every one of these is matmul-dominated
              ▼
       compute core: matmul kernels                           M0 ✅
        └─ naive │ cache-blocked │ register-blocked AVX2+FMA
              │
              ▼
        weights: fp32 → INT8/INT4                             M5
```

Milestone 0 built the bottom layer and measured it: **2.98 → 144.23 GFLOP/s**
(48× faster than the textbook loop). See [docs/05](05-kernels.md).

Read the [architecture doc](02-architecture.md) next, or
[getting started](01-getting-started.md) to run it.
