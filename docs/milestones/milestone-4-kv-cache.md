# Milestone 4 — KV cache

**Goal:** make generation practical — O(n) per token instead of O(n²) — and
learn why decode is a *different* performance problem from prefill.

## Concepts

Prefill vs decode, cached keys/values, arithmetic intensity collapsing to
matrix×vector, and memory growth with context length.

## Tasks

- [ ] **Cache structure**: preallocate `[layer][head][max_seq][head_dim]` for K
      and V at load time — never reallocate during generation.
- [ ] **Prefill path**: process the whole prompt in one pass, filling the cache
      (matrix×matrix — uses milestone 0's GEMM kernels).
- [ ] **Decode path**: for each new token compute K,V for one position, append,
      and attend against the whole cache (matrix×**vector**).
- [ ] **A `matvec` kernel tuned for decode**: it is bandwidth-bound, so the
      optimisation is streaming weights efficiently, not register blocking.
      Measure it separately from GEMM.
- [ ] **Position tracking**: RoPE must use the absolute position of each token;
      an off-by-one here degrades output subtly rather than obviously.
- [ ] **Differential test**: generation *with* the cache must produce
      **identical** token ids to generation *without* it (recomputing every
      step). This is the milestone's core proof.
- [ ] Report prefill tokens/sec and decode tokens/sec **separately** — they are
      different regimes and a single number hides the story.

## Files

`include/kvcache.hpp`, `src/kvcache.cpp`, `src/generate.cpp`,
`src/matvec.cpp`, `docs/07-kv-cache.md`.

## Definition of Done

- [ ] Cached and uncached generation produce identical outputs for several
      prompts (the differential test passes).
- [ ] Decode time per token is roughly constant with context length, not
      growing quadratically — demonstrated with a measured table.
- [ ] Prefill and decode tokens/sec reported separately, with conditions.
- [ ] Cache memory usage is reported and matches the formula in
      [docs/07](../07-kv-cache.md).
- [ ] `make all` warning-free; `make test` green.

## Notes

Expect decode to be *much* slower per FLOP than prefill — that is correct and
expected, not a bug. Reading every weight to produce one token means arithmetic
intensity is near 1, so the machine is limited by RAM bandwidth. That measurement
is exactly what motivates milestone 5.

**Next:** [Milestone 5 — Quantisation](milestone-5-quantisation.md).
