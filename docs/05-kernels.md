# 05 — Kernels

Matrix multiplication is where a transformer spends most of its time, so it is
where the engine starts. This doc explains the three implementations in
`src/matmul.cpp` and the measured result.

## The measurement (512³, one core, `-O3 -march=native`)

| kernel | ms | GFLOP/s | speedup |
|---|---|---|---|
| naive | 90.09 | 2.98 | 1.0× |
| blocked | 4.73 | 56.77 | 19.1× |
| register-blocked AVX2+FMA | 1.86 | 144.23 | 48.4× |

## 1. Naive — correct and slow

```cpp
for i, for j:
    sum = 0
    for k: sum += A[i*K+k] * B[k*N+j]
    C[i*N+j] = sum
```

The inner loop strides through `B` by `N` floats per step, so nearly every
iteration touches a different cache line. The CPU spends its time waiting for
memory, not multiplying: **3 GFLOP/s on a machine capable of ~100+**.

## 2. Cache blocking — 19× from memory access alone

Two changes, no SIMD:

- **Loop reorder to (i, k, j).** Hoist `A[i][k]` out of the inner loop and walk
  `B[k][j]` and `C[i][j]` *contiguously* along `j`. Now each cache line fetched
  is fully used.
- **Tiling.** Process the matrices in blocks (64×256×128 here) so the working
  set of each tile stays in L1/L2 instead of being evicted before reuse.

That is the single biggest lesson of the project: **19× before a single
intrinsic**, purely from touching memory in the right order.

## 3. Register blocking — the part that beats the compiler

Naively adding AVX2 intrinsics to the blocked kernel gains **nothing** — with
`-march=native` the compiler already auto-vectorises that loop. (Measured: 51
vs 54 GFLOP/s — the intrinsic version was marginally *slower*.)

The real win is holding the output tile in registers:

- A **4×16 tile of C** lives in **8 YMM accumulators** for the entire `k` loop.
- Register budget: 8 accumulators + 2 `B` vectors + 4 broadcast `A` values =
  14 of the 16 YMM registers — deliberately just under the limit, because
  spilling would undo the gain.
- `C` is loaded and stored **once per tile** instead of once per `k` iteration.

That converts a memory-bound loop into a compute-bound one: **144 GFLOP/s**,
2.5× beyond the compiler's own vectorisation and close to this core's practical
AVX2+FMA peak.

```cpp
for (k...) {
    b0 = load(B + k*N + j); b1 = load(B + k*N + j + 8);
    av = broadcast(A[(i+0)*K + k]);
    acc00 = fma(av, b0, acc00);  acc01 = fma(av, b1, acc01);
    ... rows i+1, i+2, i+3 ...
}
```

## Rules for adding a kernel

1. **Write the reference first** and keep it forever — it is the oracle.
2. **Verify before you benchmark.** `make test` compares every kernel against
   the reference at several sizes, including non-multiples of the vector width
   (1, 7, 8, 9, 64, 96) so edge handling is exercised.
3. Use a **tolerance**, never exact equality: float accumulation reorders, so
   differences around 1e-6 are expected and fine.
4. **Handle the tails.** Real shapes are not multiples of 4 or 16; the
   micro-kernel falls back to an edge routine for leftover rows/columns.
5. Guard intrinsics behind `__AVX2__`/`__FMA__` with a scalar fallback.

## Where the remaining performance is

- **Threading** (milestone 6): 4 cores ≈ 4× more, with the tiles split by rows.
- **Packing**: copying tiles of `A`/`B` into contiguous scratch buffers before
  the micro-kernel improves TLB and cache behaviour at larger sizes — the next
  step BLAS libraries take.
- **Quantisation** (milestone 5): INT8 doubles the useful memory bandwidth and
  enables wider vector ops.

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*
- [How To Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
