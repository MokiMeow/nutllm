# Milestone 2 — Transformer + tokenizer

**Goal:** a working decoder block and a BPE tokenizer — the model architecture,
running on synthetic weights.

## Concepts

Multi-head attention, causal masking, residual connections, tensor layouts and
strides, and byte-pair encoding.

## Tasks

- [ ] **Attention**: `Q = x·Wq`, `K = x·Wk`, `V = x·Wv`;
      `scores = Q·Kᵀ / sqrt(head_dim)`; causal mask (set the upper triangle to
      **−∞ before** softmax, not 0 after); `softmax`; `out = weights·V`;
      output projection `·Wo`.
- [ ] **Multi-head layout**: heads are a reshape/stride, not new math. Document
      the exact layout `[batch][head][seq][dim]` and stick to it.
- [ ] **Decoder block**: `x + attention(rmsnorm(x))`, then
      `x + swiglu_ffn(rmsnorm(x))`. Stack N of them.
- [ ] **BPE tokenizer**: load a vocab + merges file; encode text → token ids and
      decode back. Round-trip must be lossless for ASCII and UTF-8.
- [ ] **Config struct**: layers, heads, dim, head_dim, ffn_dim, vocab size,
      max_seq — read from a file in milestone 3.
- [ ] **Test with synthetic weights**: fixed-seed random weights, assert the
      forward pass is finite, shape-correct, and deterministic.
- [ ] **Reference fixture**: check in a small tensor dump (generated offline with
      PyTorch — a *fixture*, never a dependency) and assert one layer's output
      matches within tolerance.

## Files

`include/model.hpp`, `src/attention.cpp`, `src/block.cpp`,
`src/tokenizer.cpp`, `tests/fixtures/`, `docs/06-transformer.md`.

## Definition of Done

- [ ] A forward pass through N blocks runs and produces finite, deterministic
      output for fixed weights.
- [ ] Tokenizer round-trips text losslessly, including multi-byte UTF-8.
- [ ] One layer matches the reference fixture within tolerance.
- [ ] Causal masking verified: position *i* is provably unaffected by inputs at
      positions > *i* (perturb a later token, assert earlier outputs unchanged —
      this catches mask bugs that eyeballing never will).
- [ ] `make all` warning-free; `make test` green.

## Notes

The causal-mask perturbation test is the highest-value test in this milestone.
A broken mask leaks future tokens, the model still produces text, and the bug is
invisible until you wonder why generation is oddly good on the training set.

**Next:** [Milestone 3 — Real weights](milestone-3-weights.md).
