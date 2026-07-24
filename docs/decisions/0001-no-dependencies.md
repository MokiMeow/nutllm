# ADR 0001 — No dependencies (no BLAS, no PyTorch, no ONNX)

**Status:** accepted · **Date:** 2026

## Context

An inference engine needs fast linear algebra. The obvious options are OpenBLAS
/ MKL for GEMM, or building on PyTorch/ONNX Runtime/GGML entirely.

## Decision

Write everything — kernels, tensor ops, transformer, tokenizer, quantisation —
in C++17 with **no third-party libraries**.

## Rationale

- Calling `cblas_sgemm` teaches nothing about why it is fast. The knowledge this
  project exists to demonstrate — cache blocking, register blocking, FMA,
  arithmetic intensity — is precisely what a BLAS library hides.
- Milestone 0 already justifies the decision empirically: the measured path from
  2.98 → 56.77 → 144.23 GFLOP/s *is* the lesson, and it only exists because the
  kernels are ours.
- Zero dependencies means the repo builds anywhere with `g++ && make`, which
  matters for a portfolio project someone might clone and run in 30 seconds.
- llama.cpp made the same call and is the reference point for the whole
  category.

## Consequences

- More code and more responsibility: correctness must be gated by our own
  reference implementations ([docs/09](../09-testing-and-benchmarking.md)).
- We will likely be slower than a mature BLAS at large sizes. That is acceptable
  and will be reported honestly — being in the same league from scratch is the
  achievement.
- Test fixtures may be *generated* with PyTorch offline and checked in as static
  data, but PyTorch is never a build or runtime dependency.
