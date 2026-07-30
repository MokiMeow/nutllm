# 06: The transformer

*Milestones 1–2.* Concept reference; build steps are in the milestone specs.

## What a decoder block actually is

Modern decoder-only LLMs (Llama-style) repeat one block N times:

```
x ──┬─────────────────────────────────┐
    │  RMSNorm ─► attention ─────────►(+)──┬──────────────────────────┐
    └─────────────────────────────────┘    │  RMSNorm ─► SwiGLU FFN ─►(+)─► out
                                           └──────────────────────────┘
```

Two sublayers, each wrapped in a residual connection and preceded by a
normalisation. That is the whole architecture: the rest is detail.

## Attention

```
Q = x·Wq     K = x·Wk     V = x·Wv          (three matmuls  -> M0)
scores = Q·Kᵀ / sqrt(head_dim)              (matmul         -> M0)
scores = causal_mask(scores)                 (elementwise)
weights = softmax(scores)                    (M1)
out = weights·V                              (matmul         -> M0)
out = out·Wo                                 (matmul         -> M0)
```

Points that bite:

- **Causal masking**: position *i* may only attend to *j ≤ i*. Set the upper
  triangle to −∞ *before* softmax (not 0 after: that would renormalise wrongly).
- **Multi-head layout**: the head dimension is a reshape, not new math. Getting
  the strides wrong produces plausible-looking garbage: test against a
  reference on a tiny case.
- **Grouped-query attention**: there may be fewer K/V heads than query heads.
  Each contiguous group of query heads shares one KV head; TinyLlama uses
  32 query heads and 4 KV heads.
- **The 1/√head_dim scale** keeps logits in a sane range; forgetting it makes
  softmax saturate.

## Softmax, numerically

Always subtract the row max before exponentiating:

```
m = max(x);  e = exp(x - m);  out = e / sum(e)
```

Without it, `exp` overflows to `inf` for logits above ~88 in fp32 and you get
NaNs. This is the single most common numerical bug in a from-scratch engine.

## RMSNorm

Simpler than LayerNorm: no mean subtraction, no bias:

```
out = x / sqrt(mean(x²) + eps) * weight
```

Cheaper and what Llama-family models use.

## RoPE (rotary position embeddings)

Position is encoded by *rotating* pairs of dimensions in Q and K by an angle
proportional to the position. No learned position table; relative position falls
out of the dot product naturally. Applied to Q and K only: never to V.

## SwiGLU feed-forward

```
out = (silu(x·W1) * (x·W3)) · W2      where silu(z) = z * sigmoid(z)
```

Three matmuls instead of two, and the hidden dimension is typically ~2.7× the
model dimension, which is why the FFN, not attention, dominates the FLOPs at
short sequence lengths.

## Where the time goes

For a small model at short context, **the matmuls are ~90%+ of the work**, which
is why milestone 0 optimised them first. Attention's quadratic term only
dominates at long context, and that is what flash-attention-style tiling (a
stretch goal) addresses.

## Testing discipline

Build each op with a reference version and test it in isolation before
composing. Then test the whole block against known-good outputs (e.g. compare a
single layer's output against a PyTorch dump for the same weights: used as a
*test fixture*, not a runtime dependency).

## Implemented layout and block

The in-memory layout is row-major `[batch=1][sequence][head][head_dim]`; the
documented logical view `[batch][head][sequence][head_dim]` is obtained by
strides, without copying into per-head matrices. Q, K, and V projections have
shape `[sequence, dim]`. RoPE is applied independently to every head's Q/K
slice, scores above the causal diagonal become negative infinity before
softmax, and V is never rotated.

`decoder_block` implements pre-normalized attention plus a pre-normalized
SwiGLU feed-forward residual. The incremental decoder applies the same block to
one token, writes only `kv_heads` cache entries, and maps each query head to its
shared KV head. Its optimized two-layer output agrees with the permanent
reference within `2e-5`; changing the last input row leaves every earlier
attention row bit-identical.

The tokenizer begins from raw bytes, so decoding is lossless for arbitrary
UTF-8. Vocabulary and merge files use hex-encoded byte strings, which also
represents whitespace and NUL unambiguously.

## Implemented tensor-op contract

`include/ops.hpp` exposes the permanent reference and optimised paths for
softmax, RMSNorm, SwiGLU, RoPE, residual addition, and matrix-vector multiply.
The correctness gate compares vector-width edges at 7, 8, 9, 33, and 64
elements with tolerance `2e-5`. Softmax always subtracts the row maximum, RoPE
requires an even dimension and rotates Q and K together, and dependency-free
scalar fallbacks keep the build portable when AVX2 is unavailable.
