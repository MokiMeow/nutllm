# Milestone 0: Compute core ✅ (done)

**Goal:** build and measure the matrix multiplication that everything above it
depends on.

## Concepts

Cache blocking and loop order, AVX2 + FMA, register blocking, arithmetic
intensity, and reference-based correctness testing.

## What shipped

- [x] `include/tensor.hpp`: `Matrix`: row-major, 64-byte aligned, move-only.
- [x] `src/matmul.cpp`: three kernels:
      naive (i,j,k), cache-blocked (i,k,j + 64×256×128 tiles), and a
      register-blocked AVX2+FMA micro-kernel (4×16 tile in 8 YMM accumulators)
      with scalar/edge fallbacks and `__AVX2__` guards.
- [x] `src/main.cpp`: correctness harness (every kernel vs the naive reference
      at n = 1, 7, 8, 9, 64, 96) followed by the benchmark.
- [x] `make test` correctness gate; `make run` / `make bench`.

## Measured result

512³, single-threaded, best of 5, `-O3 -march=native`:

| kernel | ms | GFLOP/s | speedup |
|---|---|---|---|
| naive | 90.09 | 2.98 | 1.0× |
| blocked | 4.73 | 56.77 | 19.1× |
| register-blocked AVX2+FMA | 1.86 | **144.23** | **48.4×** |

## Definition of Done

- [x] `make clean && make all`: zero warnings.
- [x] Every kernel agrees with the naive reference (max diff ~1e-6, tolerance
      1e-3) at all six sizes, including non-multiples of 8 and of the 4×16 tile.
- [x] `make test` exits 0 and fails loudly on any disagreement.
- [x] The benchmark table is program output, with conditions stated.

## What was learned (worth keeping)

Adding AVX2 intrinsics to the blocked kernel gained **nothing**: `-march=native`
already auto-vectorised it (51 vs 54 GFLOP/s, i.e. marginally worse). The win
came only from **register blocking**: holding the 4×16 output tile in registers
across the whole `k` loop so `C` is stored once per tile rather than once per
iteration. That is the difference between vectorising the arithmetic and fixing
the memory traffic.

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*
- [docs/05: Kernels](../05-kernels.md)

**Next:** [Milestone 1: Tensor ops](milestone-1-tensor-ops.md).
