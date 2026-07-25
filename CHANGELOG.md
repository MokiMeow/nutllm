# Changelog

All notable changes to nutllm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Milestone 6 engineering pass: dependency-free row-parallel GEMM and matvec
  kernels, awkward-row/thread-count correctness coverage, reproducible scaling
  tables with efficiency, a warning-as-error build, and a checked-in CLI demo.
- Milestone 5 kernel layer: blocked symmetric INT8/INT4 quantisation with fp16
  scales, explicit low-nibble-first signed INT4 packing, in-loop dequantized
  matvec/matmul, and a stable perplexity calculator.
- Theoretically bounded round-trip and matmul differential tests across odd
  dimensions and block boundaries, byte-exact packing checks, storage accounting,
  and a deterministic fp32/INT8/INT4 synthetic perplexity comparison.
- Milestone 4: a fully preallocated `[layer][head][position][head_dim]` KV
  cache, matrix-shaped prompt prefill, streaming incremental decode, absolute
  RoPE position tracking, and cached generation as the CLI default.
- Randomized hidden-state and four-prompt token-ID differential tests against
  full-prefix recomputation, exact cache-size validation, and separate measured
  prefill/decode reporting across context lengths.
- Milestone 3 engine path: a read-only memory-mapped F32 safetensors parser,
  strict tensor/config validation, conventional Llama weight-name mapping and
  explicit layout transposition, autoregressive generation, deterministic
  greedy and seeded temperature/top-p sampling, and a model CLI.
- A reproducible two-layer synthetic checkpoint and byte-BPE fixture that
  generates `Hi!`, plus CI coverage for regeneration, malformed-file rejection,
  and command-line generation without downloads.
- Milestone 2: validated model configuration, row-major multi-head causal
  attention, pre-normalized decoder blocks, deterministic layer stacks, and a
  byte-level BPE tokenizer with hex-encoded vocabulary/merge files.
- Transformer proofs covering a checked-in identity-attention fixture,
  future-token perturbation (causal mask), two-layer optimized/reference
  agreement, finite deterministic output, and lossless ASCII/UTF-8 tokenization.
- Milestone 1: stable row-wise softmax, RMSNorm, SiLU/SwiGLU, RoPE for query
  and key, residual add, and a decode-shaped matvec path, each kept beside a
  scalar/double-precision reference.
- Correctness fixtures at vector-width edges (7, 8, 9, 33, 64), large-logit
  softmax stability, and hand-computed RMSNorm, SwiGLU, RoPE, and residual
  checks, verified with and without AVX2.
- Milestone 0: the compute core — an aligned, move-only `Matrix`, and three
  matmul kernels (naive, cache-blocked, and a register-blocked AVX2+FMA
  micro-kernel holding a 4×16 output tile in 8 YMM accumulators), with scalar
  fallbacks and `__AVX2__`/`__FMA__` guards.
- A correctness harness that verifies every kernel against the naive reference
  at sizes 1, 7, 8, 9, 64, 96 (deliberately including non-multiples of the
  vector width and the micro-kernel tile) before any timing is reported.
- A benchmark driver reporting ms, GFLOP/s, and speedup with stated conditions.
  Measured on one core at 512³: **2.98 → 56.77 → 144.23 GFLOP/s (48.4×)**.
- Build system (`make all` / `run` / `test` / `bench` / `clean`) with a portable
  `ARCH` override, and CI that gates on a warning-free build plus correctness.
- Documentation set under `docs/` (kernels, transformer, KV cache, quantisation,
  testing/benchmarking, glossary), 3 ADRs, 7 milestone specs, and the
  `AGENTS.md` operating manual.

## [0.1.0] — milestone 0
- First working version: a correctness-gated, benchmarked matmul compute core.
