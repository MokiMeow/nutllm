# Milestone 6 — Polish (portfolio pass)

**Goal:** multi-core performance, an honest head-to-head benchmark, and a repo
that impresses on sight. Tag `v1.0.0`.

## Tasks

### Performance
- [x] **Threading**: parallelise GEMM by splitting the output rows across
      threads (`std::thread`, no OpenMP dependency). Measure scaling on 2/4/N
      cores and report the efficiency, not just the speedup.
- [ ] **Thread the decode path** too, and confirm it scales worse than prefill —
      that is the expected bandwidth-bound result and worth showing.
- [ ] Optional: **packing** tiles of A/B into contiguous scratch buffers before
      the micro-kernel (the next step real BLAS takes) — measure whether it
      helps at your sizes.

### Proof
- [x] Full test suite: kernels, ops, tokenizer round-trip, causal mask,
      KV-cache differential, quantisation round-trip.
- [x] CI: build warning-free, run the correctness gate and the tiny-checkpoint
      generation test. Do **not** download models in CI.

### The headline benchmark
- [ ] **tokens/sec vs llama.cpp** — same model, same quantisation, same thread
      count, same machine. State the CPU, RAM, model, quant, and thread count.

      | engine | model | quant | threads | prefill tok/s | decode tok/s |
      |---|---|---|---|---|---|

- [ ] **Report the result honestly even if nutllm loses.** llama.cpp has years
      of tuning; being within a small factor from scratch is the achievement, and
      an explanation of *where* the gap comes from (packing, better kernels,
      threading) is more impressive than a fake win.

### Presentation
- [x] An asciinema/GIF of the engine generating text at the top of the README.
- [x] A short "where the time goes" section: prefill vs decode, and the
      kernel-optimisation ladder from milestone 0.

### Hygiene
- [ ] All status tables accurate; every milestone DoD ticked.
- [ ] `CHANGELOG.md` moved from Unreleased to `1.0.0`.
- [ ] Tag `v1.0.0`.

## Definition of Done

- [ ] Generates coherent text from a real model at a measured, published
      tokens/sec, compared against llama.cpp.
- [ ] CI green; README opens with the demo and the numbers.
- [ ] `v1.0.0` tagged.

Implementation status: row-parallel GEMM/matvec, full local/CI correctness
coverage, measured scaling, and presentation are complete. Full-model decode
integration, the same-model llama.cpp comparison, and `v1.0.0` remain open
behind milestone 3's stock-model adapter. The project deliberately does not
convert standalone kernel timings into a release claim.

## Stretch goals (after v1.0.0)

- **CUDA backend** (free GPUs on Colab/Kaggle) — the natural next tier.
- Flash-attention-style fused attention for long context.
- Speculative decoding with a small draft model.
- ARM NEON kernels (runs on a Mac / phone).
