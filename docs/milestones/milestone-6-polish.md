# Milestone 6 — Polish (portfolio pass)

**Goal:** multi-core performance, an honest head-to-head benchmark, and a repo
that impresses on sight. Tag `v1.0.0`.

## Tasks

### Performance
- [x] **Threading**: parallelise GEMM by splitting the output rows across
      threads (`std::thread`, no OpenMP dependency). Measure scaling on 2/4/N
      cores and report the efficiency, not just the speedup.
- [x] **Thread the decode path** too, and compare its scaling with prefill.
      On the measured short-prompt run decode scaled 3.14× and prefill 2.90×;
      the expected worse-decode result did not appear at this workload.
- [ ] Optional: **packing** tiles of A/B into contiguous scratch buffers before
      the micro-kernel (the next step real BLAS takes) — measure whether it
      helps at your sizes.

### Proof
- [x] Full test suite: kernels, ops, tokenizer round-trip, causal mask,
      KV-cache differential, quantisation round-trip.
- [x] CI: build warning-free, run the correctness gate and the tiny-checkpoint
      generation test. Do **not** download models in CI.

### The headline benchmark
- [x] **tokens/sec vs llama.cpp** — same model family/generation, 4-bit
      quantisation class, same thread
      count, same machine. State the CPU, RAM, model, quant, and thread count.

      | engine | model | quant | threads | prefill tok/s | decode tok/s |
      |---|---|---|---|---|---|

- [x] **Report the result honestly even if nutllm loses.** llama.cpp has years
      of tuning; being within a small factor from scratch is the achievement, and
      an explanation of *where* the gap comes from (packing, better kernels,
      threading) is more impressive than a fake win.

### Presentation
- [x] An asciinema/GIF of the engine generating text at the top of the README.
- [x] A short "where the time goes" section: prefill vs decode, and the
      kernel-optimisation ladder from milestone 0.

### Hygiene
- [x] All status tables accurate; every milestone DoD ticked.
- [x] `CHANGELOG.md` moved from Unreleased to `1.0.0`.
- [x] Tag `v1.0.0`.

## Definition of Done

- [x] Generates coherent text from a real model at a measured, published
      tokens/sec, compared against llama.cpp.
- [x] CI green; README opens with the demo and the numbers.
- [x] `v1.0.0` tagged.

Implementation status: complete. The release proof uses TinyLlama 1.1B Chat
v0.2, four WSL2 vCPUs, a 5-token prompt, and 16 generated tokens. nutllm INT4
measured 19.142 prefill and 17.089 decode tok/s; llama.cpp Q4_0 measured
149.68 and 44.43 tok/s. The formats are both 4-bit but not byte-identical:
nutllm deliberately retains fp32 embeddings and classifier, so its runtime
image is 1020.15 MiB versus the 606.54 MiB GGUF. See
[docs/09](../09-testing-and-benchmarking.md).

## Stretch goals (after v1.0.0)

- **CUDA backend** (free GPUs on Colab/Kaggle) — the natural next tier.
- Flash-attention-style fused attention for long context.
- Speculative decoding with a small draft model.
- ARM NEON kernels (runs on a Mac / phone).
