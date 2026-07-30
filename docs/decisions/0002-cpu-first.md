# ADR 0002: CPU first, GPU as a later tier

**Status:** accepted · **Date:** 2026

## Context

"LLM inference engine" suggests CUDA. But GPU development needs an NVIDIA GPU
to build and verify against, and the interesting *concepts* (arithmetic
intensity, cache/memory hierarchy, quantisation, KV caching) are architecture-
independent.

## Decision

Build the engine **CPU-first with AVX2/FMA**, and treat a **CUDA backend as a
stretch goal** after `v1.0.0`.

## Rationale

- Every milestone must be verifiable on the development machine. A CPU-first
  engine can be built, tested, and benchmarked anywhere; a CUDA-first one cannot
  be verified without the hardware, which would break the project's core rule of
  proving each milestone.
- The concepts transfer directly: register blocking on CPU is shared-memory
  tiling on GPU; both are about keeping the working set close to the ALUs.
- llama.cpp took exactly this path (CPU first, GPU backends later) and it is the
  most-used local inference engine in the world: evidence the ordering is sound.
- Free GPU access (Colab, Kaggle) exists for the stretch goal without cost or a
  credit card, so the GPU tier is not blocked, merely sequenced.

## Consequences

- The headline benchmark is CPU tokens/sec vs llama.cpp CPU: a fair,
  reproducible comparison.
- Kernels are written behind a narrow interface (`matmul.hpp`) so a CUDA
  implementation is an additional backend, not a rewrite.
- The project must be explicit that it is CPU inference, so the benchmark is not
  mistaken for a GPU claim.
