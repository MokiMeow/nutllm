# 09: Testing & benchmarking

Numerical code fails quietly: a wrong kernel returns plausible numbers rather
than crashing. These are the practices that catch it.

## The correctness gate

`make test` runs every kernel against the naive reference at sizes
**1, 7, 8, 9, 64, 96**: deliberately including non-multiples of the vector
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

1. **State the conditions**: size, run count, thread count, compiler flags, CPU.
2. **Best of N**, not a single run: minimises scheduler and frequency noise.
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
  (e.g. from PyTorch) used as a static test fixture: never as a runtime
  dependency.
- **M4**: generation with the KV cache must produce byte-identical tokens to
  generation without it.
- **M5**: quantised-vs-fp32 differential tests plus a perplexity table.
- **M6**: tokens/sec vs llama.cpp, same model, same quantisation, same machine,
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
not superlinear algorithmic scaling. These are standalone kernel results.

## Packed-panel experiment

The optional `matmul_packed` path copies bounded A and B panels into reusable
thread-local buffers, then executes an AVX2/FMA inner loop. It passes the same
tail-heavy correctness sizes as every other kernel.

| size | blocked GFLOP/s | SIMD GFLOP/s | packed GFLOP/s |
|---:|---:|---:|---:|
| 512³ | 55.50 | 142.73 | 40.67 |
| 1024³ | 49.01 | 121.59 | 53.22 |

Packing did not beat the existing register-blocked SIMD kernel. The production
path therefore remains unchanged, while the completed experiment documents why
copying panels alone is insufficient.

## TinyLlama release benchmark

Conditions:

- 13th Gen Intel Core i7-13650HX exposed as 4 WSL2 vCPUs (2 cores/4 threads),
  3.8 GiB RAM and 2.0 GiB swap.
- GCC C++17, `-O3 -march=native`, AVX2/FMA.
- TinyLlama 1.1B Chat v0.2, a 5-token prompt and 16 generated tokens.
- nutllm symmetric INT4 (block 32, fp16 scale, fp32 embeddings/classifier).
- official llama.cpp release build 10194, commit `e1a1abb78`, and the official
  TinyLlama v0.2 Q4_0 GGUF.

| engine | model storage | threads | prefill tok/s | decode tok/s |
|---|---:|---:|---:|---:|
| nutllm INT4 | 1,020.15 MiB | 1 | 6.611 | 5.444 |
| nutllm INT4 | 1,020.15 MiB | 4 | 19.142 | 17.089 |
| llama.cpp Q4_0 | 606.54 MiB | 1 | 43.10 ± 4.28 | 24.42 ± 1.09 |
| llama.cpp Q4_0 | 606.54 MiB | 4 | 149.68 ± 6.84 | 44.43 ± 2.09 |

The four-thread nutllm run scales 2.90× in prefill and 3.14× in decode. At this
short context decode did not scale worse than prefill, contrary to the initial
expectation; both remain well below llama.cpp. llama.cpp is 7.82× faster in
prefill and 2.60× faster in decode at four threads. The likely gaps are its
smaller end-to-end format, persistent/finer-grained scheduling, packed kernels,
and years of architecture-specific tuning.

This is a same model family/generation, 4-bit class, thread count, token count,
and machine comparison, not a byte-identical quantization comparison. nutllm's
policy keeps two large sensitive tensors fp32, while llama.cpp Q4_0 stores a
606.54 MiB mixed-format GGUF. That size difference is part of the result.

### Reproduction and provenance

`scripts/benchmark-tinyllama.sh` verifies all three model hashes, clean-builds
nutllm, captures one- and four-thread runs, and invokes `llama-bench`:

```bash
./scripts/benchmark-tinyllama.sh \
  /path/to/tl-chat.bin /path/to/tokenizer.bin \
  /path/to/llama-bench /path/to/ggml-model-q4_0.gguf
```

| artifact | SHA-256 |
|---|---|
| llama2.c model | `6d12ab6e18a5c1216c16053fc20647a6438fca483e4586271306e13b082213f4` |
| llama2.c tokenizer | `e610c22d05d092569bafcc2e3f9795b9b43c829fab53d3489d454614cc5b87ce` |
| official Q4_0 GGUF | `3849e8024b234f2ec0f2e3b5b59ea368804486563394886ae03d0a67ae70d504` |

The GGUF comes from the
[official TinyLlama v0.2 GGUF repository](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF);
the comparison binary comes from the
[official llama.cpp release](https://github.com/ggml-org/llama.cpp/releases).
Model downloads remain deliberately outside CI.
