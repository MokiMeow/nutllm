<h1 align="center">nutllm</h1>

<p align="center">
  <em>An LLM inference engine built from the compute up — hand-optimised
  kernels, a transformer written from scratch, and quantised weights.
  No PyTorch, no BLAS, no framework.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/lang-C%2B%2B17-orange" alt="C++17">
  <img src="https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-red" alt="AVX2">
  <img src="https://img.shields.io/badge/deps-none-brightgreen" alt="no dependencies">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
</p>

---

## What this is

Everyone can *call* an LLM. Far fewer can make one **run fast**. nutllm is the
machinery underneath: the matrix kernels, the attention math, the KV cache, and
the quantisation that let a language model produce tokens on ordinary hardware.
It is written from scratch in C++17 with no dependencies — not even BLAS.

The project starts where inference actually spends its time: **matrix
multiplication**.

## Milestone 0 result — the compute core

Same mathematics, three implementations, measured on one core:

| kernel | 512³ matmul | GFLOP/s | speedup |
|---|---|---|---|
| naive triple loop | 90.09 ms | 2.98 | 1.0× |
| cache-blocked (loop reorder + tiling) | 4.73 ms | 56.77 | **19.1×** |
| register-blocked AVX2 + FMA micro-kernel | 1.86 ms | **144.23** | **48.4×** |

*Single-threaded, best of 5 runs, `-O3 -march=native`. Every kernel is verified
against the naive reference before timing — a fast matmul that is subtly wrong
would poison every layer above it.*

**Why the last row is the interesting one.** With `-march=native` the compiler
already auto-vectorises the blocked loop, so simply adding AVX2 intrinsics buys
nothing. The win comes from **register blocking**: a 4×16 tile of the output is
held in 8 YMM accumulators across the entire `k` loop, so `C` is loaded and
stored once per tile instead of once per iteration. That turns a memory-bound
loop into a compute-bound one — 2.5× beyond what the compiler managed alone.

```bash
make run     # correctness checks + the table above
make bench   # 1024x1024
```

## Why it is interesting (the depth on show)

- **Kernels, not calls** — cache blocking, loop-order effects, FMA, and
  register blocking, each measured rather than asserted.
  ([docs/05-kernels.md](docs/05-kernels.md))
- **A transformer you can read** — attention, RMSNorm, SwiGLU, RoPE implemented
  directly from the math. ([docs/06-transformer.md](docs/06-transformer.md))
- **Memory is the real constraint** — the KV cache and why generation is
  bandwidth-bound, not FLOP-bound.
  ([docs/07-kv-cache.md](docs/07-kv-cache.md))
- **Quantisation** — INT8/INT4 blocked quantisation, the accuracy/size
  trade-off, and dequantising inside the kernel.
  ([docs/08-quantisation.md](docs/08-quantisation.md))

## Status

Milestones 0 through 2 are complete. Milestone 3's memory-mapped loader,
generation loop, sampling, CLI, and tiny-checkpoint CI proof are complete; a
stock open-model adapter and coherent-text validation remain. The road to a
real model is in [docs/04-roadmap.md](docs/04-roadmap.md).

| # | Milestone | State |
|---|-----------|-------|
| 0 | Compute core: matmul kernels + benchmark | ✅ done |
| 1 | Tensor ops: softmax, RMSNorm, SwiGLU, RoPE | ✅ done |
| 2 | Transformer block + tokenizer (BPE) | ✅ done |
| 3 | Safetensors + generation; stock-model validation | 🟡 partial |
| 4 | KV cache + incremental decoding | ✅ done |
| 5 | INT8/INT4 quantisation | ⬜ |
| 6 | Threading, benchmarks vs llama.cpp, `v1.0.0` | ⬜ |

The endgame: **run a real open-weights model and publish tokens/sec** against
llama.cpp on the same machine.

## Quick start

```bash
sudo apt-get install -y g++ make   # one time
make run                           # verify + benchmark
make test                          # correctness gate (what CI runs)
./build/nutllm 1024                # benchmark a chosen size
./build/nutllm --model tests/fixtures/tiny.safetensors \
  --vocab tests/fixtures/tiny.vocab --merges tests/fixtures/tiny.merges \
  --prompt H --max-tokens 8        # prints Hi!
```

Requires x86-64 with AVX2 + FMA (any CPU since ~2013). Without them the SIMD
kernel falls back to the blocked version and says so.

## Repository layout

```
nutllm/
├── src/          # matmul/tensor kernels, self-tests, benchmark driver
├── include/      # Matrix and kernel declarations
├── docs/         # kernels, transformer, KV cache, quantisation, roadmap
└── Makefile      # all / run / test / bench / clean
```

## License

MIT — see [LICENSE](LICENSE).
