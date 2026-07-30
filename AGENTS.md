# AGENTS.md: how this repo is built

The working agreement for this repository: anyone contributing to nutllm should
read it fully before making changes. If anything here conflicts with a note elsewhere, **this file
wins.**

---

## 1. How the work is organised

- **Planning**: plans milestones, defines Definitions of Done, reviews
  diffs, keeps benchmarks and docs honest.
- **Implementation**: proceed one milestone at a time against `docs/milestones/`,
  keeping the build clean, the correctness gate green, and the numbers real.

The loop: **pick the lowest-numbered unfinished milestone → implement →
build clean → prove correctness against a reference → measure → tick the
Definition of Done → update docs/CHANGELOG → commit → next.**

## 2. Ground rules (non-negotiable)

1. **Correctness before speed, always.** Every kernel must be verified against a
   simple reference implementation *before* it is benchmarked. A fast kernel
   that is subtly wrong poisons every layer above it and is worse than no
   kernel. `make test` is the gate.
2. **No dependencies.** C++17 standard library and SIMD intrinsics only. No
   BLAS, no Eigen, no PyTorch, no ONNX. Writing the kernels *is* the project.
3. **Never report a number you did not measure.** Benchmarks state the size, the
   run count, the threading, and the machine. No estimates, no "should be".
4. **The build must never break.** `make clean && make all` with zero warnings
   at every commit; `make test` green.
5. **New `.cpp` → `src/`, new `.hpp` → `include/`.** The Makefile globs
   `src/*.cpp`.
6. **Guard SIMD behind feature macros** (`__AVX2__`, `__FMA__`) with a scalar
   fallback, so the project still builds and runs on a machine without them.
7. **No feature without a doc.** Update the relevant `docs/` page in the same
   milestone.

## 3. Build, run, verify

| Command | What it does |
|---------|--------------|
| `make all` | Build `build/nutllm` (zero warnings expected). |
| `make run` | Correctness checks + the benchmark table. |
| `make test` | Correctness gate only: fails if any kernel disagrees. |
| `make bench` | Larger benchmark (1024). |
| `make clean` | Remove `build/`. |

Portable build (no `-march=native`): `make ARCH="-mavx2 -mfma"`.

**Definition of "it works":** build warning-free, `make test` exits 0 with no
`FAIL` lines, and any performance claim is backed by output you ran.

## 4. Coding standards

- **C++17**, 4-space indent, `snake_case` functions/variables, `PascalCase`
  types, trailing `_` on private members.
- Kernels: keep the reference version alongside the optimised one, forever. It
  is the oracle.
- Comment *why* a tile targets a cache level, why a loop order was chosen,
  what register pressure a micro-kernel has. Not what the loop does.
- Prefer plain functions and small structs over class hierarchies or templates.
- Numerical code: state tolerances explicitly; float accumulation reorders, so
  never test for exact equality across kernels.

## 5. Commit and branch style

- `type(scope): outcome`, imperative, lower case.
  Examples: `feat(kernels): add register-blocked avx2 micro-kernel`,
  `fix(attention): correct the causal mask off-by-one`.
- Types: `feat`, `fix`, `docs`, `refactor`, `build`, `chore`, `test`, `perf`.
- **No AI/co-author trailers.**
- Branch per milestone (`milestone-2-transformer`), PR into `main`, CI green.

## 6. The milestone path

Specs with Definitions of Done live in `docs/milestones/`.

| # | Milestone | Adds | Spec |
|---|-----------|------|------|
| 0 | Compute core | Matrix, 3 matmul kernels, correctness + benchmark | [spec](docs/milestones/milestone-0-compute-core.md) ✅ |
| 1 | Tensor ops ✅ | softmax, RMSNorm, SwiGLU, RoPE, all reference-tested | [spec](docs/milestones/milestone-1-tensor-ops.md) |
| 2 | Transformer ✅ | attention block, BPE tokenizer, forward pass | [spec](docs/milestones/milestone-2-transformer.md) |
| 3 | Real weights | GGUF/safetensors loader, generate text from a real model | [spec](docs/milestones/milestone-3-weights.md) |
| 4 | KV cache | incremental decoding, prefill vs decode | [spec](docs/milestones/milestone-4-kv-cache.md) |
| 5 | Quantisation | INT8/INT4 blocked quant, dequant-in-kernel | [spec](docs/milestones/milestone-5-quantisation.md) |
| 6 | Polish | threading, tokens/sec vs llama.cpp, CI, `v1.0.0` | [spec](docs/milestones/milestone-6-polish.md) |

**Definition of Done (whole project):** nutllm loads a real open-weights model,
generates coherent text, and publishes a measured tokens/sec comparison against
llama.cpp on the same machine.

## 7. What NOT to do

- Do not add a dependency to "save time": the kernels are the point.
- Do not delete or bypass the reference implementations.
- Do not report a speedup without the measurement that produced it.
- Do not optimise before the correctness gate passes for the new code.
- Do not hard-code a model path or download weights in CI (large files; keep the
  loader tested with a tiny synthetic checkpoint instead).

## 8. Tools reference

- **g++** (C++17, `-O3 -march=native`).
- **Intel Intrinsics Guide**: <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/>
 : the authority for every intrinsic used.
- **`perf stat`**: cache misses and IPC when a kernel underperforms.
- **`objdump -d`**: confirm the compiler actually emitted `vfmadd*` where you
  expected.

Build one milestone, prove it, measure it, document it, commit. Then the next.
