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

<p align="center">
  <img src="docs/assets/demo.svg"
       alt="nutllm loads the tiny checkpoint and generates Hi!">
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

Version 1.0.0 completes the full roadmap: real TinyLlama weights, grouped-query
attention, cached decoding, real INT8/INT4 generation, reusable worker threads,
and an honest llama.cpp comparison. CI keeps the complete tiny-checkpoint path
network-free.

| # | Milestone | State |
|---|-----------|-------|
| 0 | Compute core: matmul kernels + benchmark | ✅ done |
| 1 | Tensor ops: softmax, RMSNorm, SwiGLU, RoPE | ✅ done |
| 2 | Transformer block + tokenizer (BPE) | ✅ done |
| 3 | Safetensors/llama2.c + coherent real-model generation | ✅ done |
| 4 | KV cache + incremental decoding | ✅ done |
| 5 | INT8/INT4 kernels + real-model proof | ✅ done |
| 6 | Threading/CI/presentation + llama.cpp release proof | ✅ done |

## Headline result

TinyLlama 1.1B Chat v0.2, four WSL2 vCPUs, a 5-token prompt and 16 generated
tokens:

| engine | 4-bit format | model storage | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|
| **nutllm** | symmetric INT4, block 32 | 1,020.15 MiB | 19.142 | 17.089 |
| llama.cpp b10194 | Q4_0 | 606.54 MiB | 149.68 ± 6.84 | 44.43 ± 2.09 |

llama.cpp wins by 7.82× in prefill and 2.60× in decode. That is the honest
result: its GGUF is smaller and its scheduling, packing, and kernels are much
more mature. The formats are both 4-bit but not byte-identical—nutllm keeps the
embeddings and classifier fp32 by design. Full conditions, hashes, one-thread
results, and reproduction commands are in
[docs/09-testing-and-benchmarking.md](docs/09-testing-and-benchmarking.md).

## Where the time goes

Prompt prefill is matrix×matrix and compute-bound, so cache blocking, SIMD, and
row threading pay off. Incremental decode is matrix×vector plus a linear scan
of cached attention history; it streams weights and becomes bandwidth-bound.
That is why the project reports these phases separately and why INT4 storage is
useful even though unpacking adds arithmetic.

On the 4-vCPU WSL2 environment, the standalone row-parallel kernels measured:

| threads | 768³ GEMM | speedup | efficiency | 4096² matvec | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.51 ms | 1.00× | 100.0% | 10.22 ms | 1.00× |
| 2 | 13.49 ms | 2.04× | 102.0% | 5.19 ms | 1.97× |
| 4 | 9.15 ms | 3.01× | 75.2% | 2.60 ms | 3.93× |

Median timings, `-O3 -march=native`, 13th Gen Intel Core i7-13650HX exposed
through WSL2. The release benchmark above measures the full decoder separately.

## Quick start

```bash
sudo apt-get install -y g++ make   # one time
make run                           # verify + benchmark
make test                          # correctness gate (what CI runs)
./build/nutllm 1024                # benchmark a chosen size
./build/nutllm --model tests/fixtures/tiny.safetensors \
  --vocab tests/fixtures/tiny.vocab --merges tests/fixtures/tiny.merges \
  --prompt H --max-tokens 8        # prints Hi!
./build/nutllm --model /path/to/tl-chat.bin \
  --tokenizer /path/to/tokenizer.bin --prompt "Once upon a time" \
  --max-tokens 16 --quant int4 --threads 4 --stats
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
