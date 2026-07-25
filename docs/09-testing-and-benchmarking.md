# 09 — Testing & benchmarking

Numerical code fails quietly: a wrong kernel returns plausible numbers rather
than crashing. These are the practices that catch it.

## The correctness gate

`make test` runs every kernel against the naive reference at sizes
**1, 7, 8, 9, 64, 96** — deliberately including non-multiples of the vector
width (8) and the micro-kernel tile (4×16), because **edge handling is where
SIMD kernels break**. A kernel that is right at 64 and wrong at 9 has a tail-loop
bug.

Rules:

- **Tolerance, not equality.** Float addition is not associative, so reordering
  changes the last bits. `1e-3` here is far looser than the observed `~1e-6` and
  far tighter than any real bug.
- **The reference never leaves.** `matmul_naive` is permanent infrastructure.
- **Correctness runs before timing**, in the same binary, so an optimisation
  that breaks accuracy can never be reported as a speedup.

## Benchmarking honestly

The rules this project holds itself to:

1. **State the conditions** — size, run count, thread count, compiler flags, CPU.
2. **Best of N**, not a single run — minimises scheduler and frequency noise.
3. **Use the result.** If nothing reads the output matrix, the compiler may
   delete the work entirely and report absurd GFLOP/s. Here the correctness pass
   consumes it.
4. **Compare against theoretical peak** to know what is left:
   `peak ≈ FMA_units × 8 floats × 2 flops × GHz`. Milestone 0's 144 GFLOP/s sits
   near a single core's practical ceiling, which is the evidence the kernel is
   genuinely good rather than merely better.
5. **Never quote a number you did not run on your machine.** The README's table
   is program output.

## When a kernel underperforms

```bash
perf stat -e cycles,instructions,cache-misses,LLC-load-misses ./build/nutllm 1024
```

- **High cache misses** → blocking/tiling problem, not an arithmetic problem.
- **Low IPC** → dependency stalls; more accumulators may help (that is exactly
  what register blocking fixed in milestone 0).
- **No `vfmadd` in the disassembly** → the compiler did not vectorise what you
  assumed:

```bash
objdump -d build/matmul.o | grep -c vfmadd
```

## Later milestones

- **M2/M3**: compare a single transformer layer against a reference dump
  (e.g. from PyTorch) used as a static test fixture — never as a runtime
  dependency.
- **M4**: generation with the KV cache must produce byte-identical tokens to
  generation without it.
- **M5**: quantised-vs-fp32 differential tests plus a perplexity table.
- **M6**: tokens/sec vs llama.cpp, same model, same quantisation, same machine —
  and report it even if we lose.

## Milestone 6 thread scaling

Median local results on the 4-vCPU WSL2 environment (13th Gen Intel Core
i7-13650HX, `-O3 -march=native`):

| threads | 768³ GEMM ms | speedup | efficiency | 4096² matvec ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.51 | 1.00× | 100.0% | 10.22 | 1.00× |
| 2 | 13.49 | 2.04× | 102.0% | 5.19 | 1.97× |
| 4 | 9.15 | 3.01× | 75.2% | 2.60 | 3.93× |

The >100% two-thread GEMM efficiency is normal measurement/turbo variation,
not superlinear algorithmic scaling. These are standalone kernel results. They
are not substituted for the still-pending same-model llama.cpp comparison.
