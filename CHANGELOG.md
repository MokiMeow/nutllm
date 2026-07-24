# Changelog

All notable changes to nutllm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
