# ADR 0003 — Reference implementations are permanent infrastructure

**Status:** accepted · **Date:** 2026

## Context

Once a fast kernel works, the obvious cleanup is to delete the slow one. In
numerical code that is a mistake: optimised kernels fail *quietly*, returning
plausible numbers rather than crashing.

## Decision

Every optimised kernel keeps its plain reference implementation in the codebase
permanently, and `make test` checks each optimised kernel against the reference
**before** any timing is reported.

## Rationale

- Numerical bugs do not announce themselves. A tail-loop that drops the last
  three columns of a matrix still produces a matrix, and a transformer built on
  it still emits text — just slightly wrong text, which is nearly impossible to
  debug from the top.
- The reference doubles as executable documentation: `matmul_naive` is the
  clearest possible statement of what the kernel computes.
- Milestone 0's test sizes (1, 7, 8, 9, 64, 96) exist precisely to catch edge
  handling around the vector width (8) and micro-kernel tile (4×16) — the bugs
  that a "fast path only" suite would miss.
- The same discipline extends upward: M4 checks KV-cached generation against
  uncached generation; M5 checks quantised kernels against fp32.

## Consequences

- A permanent, deliberately slow code path exists in the repo — documented as
  intentional so it is not mistaken for dead code.
- Test time grows with the reference's O(n³) cost, so correctness runs at small
  sizes and benchmarks at large ones.
- Any claim of a speedup is automatically a claim about *correct* code, because
  the gate runs first in the same binary.
