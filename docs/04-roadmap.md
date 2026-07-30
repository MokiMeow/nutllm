# 04: Roadmap

From "a matmul at 144 GFLOP/s" to a real 1.1B language model with published,
reproducible tokens/sec.

## The plan

| # | Milestone | You'll build | You'll learn |
|---|-----------|--------------|--------------|
| 0 | **Compute core** ✅ | Matrix + 3 matmul kernels + benchmark | cache blocking, FMA, register blocking, why memory dominates |
| 1 | **Tensor ops** ✅ | softmax, RMSNorm, SwiGLU, RoPE, residual, matvec | numerical stability, fused ops, reference testing |
| 2 | **Transformer** ✅ | attention block, byte-level BPE tokenizer, forward pass | Q/K/V, causal masking, multi-head layout |
| 3 | **Real weights** ✅ | safetensors + llama2.c loaders, real tokenizer, coherent TinyLlama output | file formats, mmap/pread, tensor layouts |
| 4 | **KV cache** ✅ | incremental decoding | prefill vs decode; why generation is bandwidth-bound |
| 5 | **Quantisation** ✅ | INT8/INT4 kernels and coherent 1.1B generation | accuracy/size trade-offs, dequant inside the kernel |
| 6 | **Polish** ✅ | persistent threading, CI, presentation, llama.cpp comparison, v1.0.0 | honest benchmarking, presentation |

## Dependency order

```
M0 ─► M1 ─► M2 ─► M3 ─► M4 ─► M5 ─► M6
```

Linear by necessity: the transformer needs the tensor ops, real weights need the
transformer, the KV cache only matters once generation works, and quantisation
is only measurable against an unquantised baseline.

## Definition of Done (whole project)

nutllm loads a real open-weights model (a small one: 1B–3B parameters is
plenty), generates coherent text from a prompt, and publishes a **measured**
tokens/sec comparison against llama.cpp on the same machine, with the model,
quantisation level, thread count, and CPU stated.

## The headline artifact

| engine | model | quant | tokens/sec |
|---|---|---|---|
| llama.cpp | TinyLlama 1.1B Chat v0.2 | Q4_0 | 44.43 decode |
| **nutllm** | TinyLlama 1.1B Chat v0.2 | INT4 block 32 | 17.089 decode |

Both numbers are four-thread measurements on the same machine. The
quantization policies are both 4-bit but not byte-identical; the full conditions
and explanation of the 2.60× decode gap are in
[docs/09](09-testing-and-benchmarking.md).

## Stretch goals (after v1.0.0)

- **CUDA backend**: port the micro-kernel to a GPU; the natural next tier, and
  where the free Colab/Kaggle GPUs come in.
- Flash-attention-style fused attention (tiling to avoid materialising the
  full attention matrix).
- Speculative decoding with a small draft model.
- ARM NEON kernels so it runs on a Mac or a phone.
- Batched inference (multiple sequences at once).
