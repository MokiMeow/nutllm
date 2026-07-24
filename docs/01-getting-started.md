# 01 — Getting started

## Requirements

x86-64 Linux or WSL2 with `g++` (C++17) and `make`:

```bash
sudo apt-get update && sudo apt-get install -y g++ make
```

AVX2 + FMA (any CPU since roughly 2013) unlocks the fast kernel. Check:

```bash
grep -m1 -o avx2 /proc/cpuinfo && grep -m1 -o fma /proc/cpuinfo
```

Without them the build still works — the SIMD kernel falls back to the blocked
one and the output says so.

## Build and run

```bash
make run
```

Runs the correctness checks, then the benchmark:

```
nutllm milestone 0 — compute core (AVX2/FMA: yes)

correctness (each kernel vs the naive reference)
  ok   n=1    blocked_diff=0.00e+00 simd_diff=0.00e+00
  ...
benchmark  512x512 x 512x512  (best of 3/5 runs)
  kernel            ms   GFLOP/s  speedup
  naive          90.09      2.98     1.0x
  blocked         4.73     56.77    19.1x
  simd            1.86    144.23    48.4x
```

Your numbers will differ — they depend on your CPU. Always quote your own.

## Targets

| Command | Purpose |
|---------|---------|
| `make all` | Build `build/nutllm`. |
| `make run` | Correctness + benchmark at 512. |
| `make test` | Correctness gate only (CI runs this). |
| `make bench` | Benchmark at 1024. |
| `make clean` | Delete `build/`. |

`./build/nutllm <size>` benchmarks any square size.

## Portable vs native builds

`-march=native` (the default) tunes for *this* machine and may produce a binary
that crashes elsewhere with SIGILL. For a portable build:

```bash
make ARCH="-mavx2 -mfma"
```

## Reading the benchmark honestly

- It is **single-threaded**; multi-core comes in milestone 6.
- It reports the **best of N runs**, which minimises scheduler noise — say so
  whenever you quote it.
- GFLOP/s = `2·N³ / seconds`: one multiply and one add per inner iteration.
- Compare against your CPU's theoretical peak
  (`FMA units × 8 floats × 2 flops × GHz`) to know how much is left.

## Troubleshooting

- **`FAIL` in the correctness output** — a kernel disagrees with the reference.
  Fix that before looking at any timing.
- **SIGILL** — a binary built with `-march=native` run on a different CPU.
  Rebuild portable.
- **Suspiciously high GFLOP/s** — the compiler may have optimised the work away
  (results unused). The correctness check reading the output prevents this here.
