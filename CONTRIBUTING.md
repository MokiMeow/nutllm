# Contributing to nutllm

This is primarily a learning/portfolio project, but clean contributions are
welcome.

## Before you start

- Read [AGENTS.md](AGENTS.md) — the operating manual, which applies to humans
  too.
- Skim [docs/00-overview.md](docs/00-overview.md), the
  [roadmap](docs/04-roadmap.md), and [docs/05-kernels.md](docs/05-kernels.md).
- You need x86-64 Linux or WSL2 with `g++` (C++17). AVX2 + FMA recommended.

## The two rules that matter most

1. **Correctness before speed.** Write the plain reference implementation first,
   keep it forever, and verify the optimised version against it *before* timing
   anything. Numerical bugs return plausible wrong numbers instead of crashing.
2. **Never report a number you did not measure.** State the size, run count,
   thread count, compiler flags, and CPU with every benchmark.

## Workflow

1. Pick the lowest-numbered unfinished milestone in
   [docs/milestones/](docs/milestones/), or an open issue.
2. Branch from `main`: `git checkout -b milestone-2-transformer`.
3. `make clean && make all` must be warning-free and `make test` must pass at
   every commit.
4. Add correctness cases for anything new, including edge sizes (1, 7, 8, 9, 33).
5. Update the relevant doc and tick the Definition of Done.
6. Open a PR into `main`; CI must be green.

## Commit style

`type(scope): outcome` in the imperative, lower case. Types: `feat`, `fix`,
`docs`, `refactor`, `build`, `chore`, `test`, `perf`. No AI/co-author trailers.

Example: `perf(kernels): hold the output tile in ymm accumulators`.

## Code style

See §4 of [AGENTS.md](AGENTS.md). C++17, 4-space indent, `snake_case` functions,
`PascalCase` types. Comment *why* — which cache level a tile targets, what
register pressure a micro-kernel has.

## Reporting issues

Include your CPU, the output of `make run`, and — for a correctness failure —
the sizes and diffs printed by the harness.
