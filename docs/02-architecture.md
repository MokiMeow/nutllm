# 02 — Architecture

nutllm is layered so that each level depends only on the level beneath it, and
each level has a reference implementation that defines "correct."

```
┌──────────────────────────────────────────────┐
│ generation loop: sample, decode        (M3/M4)│
├──────────────────────────────────────────────┤
│ tokenizer (BPE)                           (M2)│
├──────────────────────────────────────────────┤
│ transformer: attention, FFN, blocks       (M2)│
├──────────────────────────────────────────────┤
│ tensor ops: softmax, RMSNorm, SwiGLU, RoPE(M1)│
├──────────────────────────────────────────────┤
│ compute core: matmul kernels           (M0) ✅│
├──────────────────────────────────────────────┤
│ Matrix: aligned storage                (M0) ✅│
└──────────────────────────────────────────────┘
      weights: fp32 / INT8 / INT4          (M5)
```

## Current components

| File | Role |
|------|------|
| `include/tensor.hpp` | `Matrix` — row-major, 64-byte aligned, move-only |
| `include/matmul.hpp` | fp32 kernels and the persistent row-worker executor |
| `include/llama2.hpp` | llama2.c checkpoint/tokenizer adapter |
| `include/kvcache.hpp` | preallocated GQA-aware KV cache |
| `include/quant.hpp` | blocked INT8/INT4 storage and inner-loop dequantization |
| `include/quant_model.hpp` | real quantized transformer weights and decoder |
| `src/matmul.cpp` | naive / blocked / register-blocked implementations |
| `src/main.cpp` | correctness harness, benchmark driver, and model CLI |

## Design choices that matter

**Aligned, move-only `Matrix`.** 64-byte alignment (a cache line, and a
multiple of AVX2's 32) avoids split-line penalties on every vector access.
Copying is deleted, moving is allowed: accidentally copying a weight matrix is a
performance bug, so the type makes it impossible.

**Reference implementations are permanent.** `matmul_naive` is not scaffolding —
it is the oracle every future kernel is checked against, and it stays in the
codebase forever.

**Kernels take shapes, not policies.** Blocking sizes are constants in one
place. A reusable row executor wraps GEMM, fp32 matvec, linear, and quantized
matvec work, avoiding fresh worker creation for every projection.

**Feature-guarded SIMD.** Intrinsics live behind `__AVX2__`/`__FMA__` with a
scalar fallback, so the project builds and runs anywhere and degrades honestly.

## How the layers grew

- **M1** added elementwise/reduction ops (softmax, RMSNorm, SwiGLU, RoPE), each
  with a reference and a test — same discipline as matmul.
- **M2** composed them into an attention block; attention is itself three
  matmuls plus a softmax, so it sits directly on M0.
- **M3** added safetensors and llama2.c loaders plus the sampling loop.
- **M4** added the KV cache, which changes decode from "matmul" to
  "matrix-vector" — bandwidth-bound, a different optimisation problem
  ([docs/07](07-kv-cache.md)).
- **M5** replaced projection weights with INT8/INT4 blocks, dequantised inside the
  kernel so the memory traffic actually drops
  ([docs/08](08-quantisation.md)).
- **M6** parallelised across cores and benchmarked against llama.cpp.

See the [roadmap](04-roadmap.md) and the [milestones](milestones/).
