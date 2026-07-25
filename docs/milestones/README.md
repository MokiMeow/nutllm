# Milestones

Each milestone leaves an engine that **builds clean, passes the correctness
gate, and has measured numbers**. Build them in order.

| # | Milestone | State |
|---|-----------|-------|
| 0 | [Compute core](milestone-0-compute-core.md) | ✅ done |
| 1 | [Tensor ops](milestone-1-tensor-ops.md) | ✅ done |
| 2 | [Transformer + tokenizer](milestone-2-transformer.md) | ✅ done |
| 3 | [Real weights](milestone-3-weights.md) | ⬜ |
| 4 | [KV cache](milestone-4-kv-cache.md) | ⬜ |
| 5 | [Quantisation](milestone-5-quantisation.md) | ⬜ |
| 6 | [Polish](milestone-6-polish.md) | ⬜ |

## Every milestone spec has

**Goal · Concepts · Tasks · Files · Definition of Done · References.**

## The loop (from AGENTS.md)

1. Pick the lowest-numbered unfinished milestone.
2. Write the **reference implementation first**, then the optimised one.
3. Verify against the reference (`make test`) — *before* timing anything.
4. Measure, and record the conditions (size, runs, threads, CPU).
5. Update the doc, tick the DoD, update README/CHANGELOG/roadmap.
6. Commit (`type(scope): …`), keep CI green.

## The rule that matters most

**Correctness before speed.** A kernel that is fast and subtly wrong is worse
than no kernel — every layer above it inherits the error, and the symptom
(slightly wrong text) is nearly impossible to trace back down.
