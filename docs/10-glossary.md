# 10 — Glossary

- **Arithmetic intensity** — FLOPs performed per byte loaded. High for matrix×
  matrix (prefill), very low for matrix×vector (decode).
- **Attention** — the operation that lets each token weigh earlier tokens:
  `softmax(Q·Kᵀ/√d)·V`.
- **AVX2** — 256-bit x86 SIMD: 8 floats per vector register (YMM).
- **Bandwidth-bound** — limited by how fast data streams from memory, not by
  arithmetic. Decode is bandwidth-bound.
- **BPE (byte-pair encoding)** — the tokenizer scheme mapping text to token ids.
- **Cache blocking / tiling** — processing sub-blocks so the working set fits in
  L1/L2; the source of milestone 0's first 19×.
- **Causal mask** — prevents a position attending to later positions; applied as
  −∞ before softmax.
- **Compute-bound** — limited by arithmetic throughput. Prefill is compute-bound.
- **Decode** — generating tokens one at a time after the prompt.
- **Dequantisation** — converting quantised weights back to float; must happen
  *inside* the kernel to preserve the bandwidth win.
- **FMA (fused multiply-add)** — `a*b + c` as one instruction with one rounding;
  doubles effective FLOPs per cycle.
- **GFLOP/s** — billions of floating-point operations per second;
  `2·N³/seconds` for an N³ matmul.
- **GGUF** — the file format llama.cpp uses for quantised model weights.
- **KV cache** — stored keys and values for previous tokens, so decode is O(n)
  instead of O(n²).
- **Micro-kernel** — the innermost register-blocked loop of a high-performance
  matmul; here 4 rows × 16 columns in 8 YMM accumulators.
- **Perplexity** — how well a model predicts held-out text; the metric for
  judging quantisation damage.
- **Prefill** — processing the whole prompt in one pass before generation.
- **Quantisation** — storing weights in fewer bits (INT8/INT4) with per-block
  scales.
- **Register blocking** — keeping an output tile in CPU registers across the
  inner loop, so it is not reloaded/stored every iteration. The change that took
  milestone 0 from 57 to 144 GFLOP/s.
- **RMSNorm** — normalisation by root-mean-square, no mean subtraction; used by
  Llama-family models.
- **RoPE** — rotary position embeddings; encodes position by rotating Q/K pairs.
- **SIMD** — one instruction operating on several values at once.
- **SwiGLU** — the gated feed-forward variant used in modern LLMs.
- **Tokens/sec** — the end-user metric this engine is ultimately judged on.
- **YMM register** — a 256-bit AVX register; there are 16 per core, and the
  micro-kernel deliberately uses 14.
