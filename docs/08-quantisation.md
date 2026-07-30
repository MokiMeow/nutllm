# 08 — Quantisation

*Milestone 5.* Storing weights in 8 or 4 bits instead of 32 — the largest single
win for generation speed, because decode is bandwidth-bound
([docs/07](07-kv-cache.md)).

## Why it works

A 3B-parameter model is 12 GB in fp32, 3 GB in INT4. During decode the engine
must stream every weight to produce each token, so **four times fewer bytes ≈
four times faster**, even though the arithmetic is unchanged. Quantisation is a
memory optimisation that happens to be expressed in numerics.

## Blocked (group-wise) quantisation

Quantising a whole tensor with one scale is inaccurate — a single outlier
stretches the range and crushes precision everywhere else. Instead, quantise in
small blocks (32 or 64 weights), each with its own scale:

```
for each block of 32 weights:
    scale = max(|w|) / 7          # emitted symmetric range is [-7, 7]
    q[i]  = clamp(round(w[i] / scale), -7, 7)
store: 32 × 4-bit values + one fp16 scale
```

Effective bits per weight = 4 + 16/32 = **4.5**, and accuracy stays close to
fp32 because outliers only affect their own block.

INT4 uses signed two's-complement nibbles. Flattened even-indexed weights occupy
the **low nibble** and odd-indexed weights occupy the **high nibble**. The
quantiser emits `[-7, 7]`; `0x8` (−8) is accepted by the decoder but never
created. For example `[-7, -1, 2, 7]` packs as bytes `f9 72`.

## Dequantising inside the kernel

The point is missed entirely if you dequantise a whole matrix to fp32 in memory
and then call the fp32 kernel — you would read *more* bytes than before. The
dequantisation must happen **inside** the matmul, per block, in registers:

```
load 32 packed 4-bit weights          (16 bytes)
unpack to int8 → convert to float     (in registers)
multiply by the block scale
FMA into the accumulators
```

Memory traffic drops; register work rises slightly. That trade is the whole
optimisation.

## Accuracy: measure, don't assume

Report **perplexity** on a fixed text sample for fp32 vs INT8 vs INT4 on the
same model. A typical, honest result is: INT8 nearly identical, INT4 slightly
worse but usually acceptable. Publish the numbers — a quantisation claim without
a perplexity table is meaningless.

| format | bits/weight | size (3B) | perplexity | tokens/sec |
|---|---|---|---|---|
| fp32 | 32 | 12 GB | *baseline* | X |
| INT8 | 8.5 | 3.2 GB | *measured* | Y |
| INT4 | 4.5 | 1.7 GB | *measured* | Z |

## Correctness discipline

1. **Round-trip test**: quantise then dequantise a known tensor; the max error
   must be within the block's theoretical bound.
2. **Kernel differential test**: the quantised matmul vs the fp32 reference on
   the same weights, with a tolerance derived from the quantisation error — not
   an arbitrary constant.
3. **End-to-end**: the model must still produce coherent text. Garbage output
   after quantisation usually means a packing/unpacking bug (nibble order), not
   an accuracy limit.

## Notes

- **Symmetric** quantisation (scale only, no zero-point) is simpler and fine for
  weights, which are roughly zero-centred.
- **Keep some tensors in higher precision** — embeddings and the output
  projection are sensitive; llama.cpp does the same.
- Store the nibble packing order explicitly in the docs; it is the most common
  source of "why is my output garbage."

## Measured accuracy proof

The checked-in deterministic test uses 185 weights (block size 32), including
an odd final nibble and block boundaries:

| format | stored bytes | effective bits/weight | synthetic perplexity |
|---|---:|---:|---:|
| fp32 | 740 | 32.00 | 11.598 |
| INT8 + fp16 scales | 197 | 8.52 | 11.608 |
| INT4 + fp16 scales | 105 | 4.54 | 11.715 |

The three perplexities use the same fixed logits and targets. This is a
numerical regression fixture, not a claim about TinyLlama corpus quality.
Round-trip worst-error/bound ratios were 0.879 (INT8) and 0.993 (INT4);
quantized-matmul ratios were 0.726 and 0.756. A ratio below 1 proves every
observed error stayed inside the bound derived from rounding and fp16 scale
storage.

## Real-model storage, speed, and quality

TinyLlama 1.1B Chat v0.2 was loaded from the same fp32 llama2.c checkpoint for
both formats. All seven large projections per layer are quantized; embeddings,
normalization weights, and the separate classifier stay fp32.

| format | projection bits/weight | runtime image | fixed-fixture perplexity | real decode tok/s |
|---|---:|---:|---:|---:|
| fp32 | 32.00 | 4,196.90 MiB source | 11.598 | not run in 3.8 GiB guest |
| INT8 + fp16 scales | 8.50 | 1,482.15 MiB | 11.608 | 15.388 |
| INT4 + fp16 scales | 4.50 | 1,020.15 MiB | 11.715 | 15.985 |

Real runs used four threads, a 5-token `Once upon a time` prompt, 16 generated
tokens, `-O3 -march=native`, and the machine described in
[docs/09](09-testing-and-benchmarking.md). INT8 produced the same
`What is the capital of France?` greedy prefix as the fp32 llama2.c reference;
INT4 produced `Dame Fortune, I am your guest. I am a powerful sor`.

The simple bandwidth prediction did **not** hold end to end: the INT4 runtime
image is 31% smaller than INT8, but decode improved only 1.04×. Profiling
implications are clear from the design: the fp32 classifier is streamed every
token, and INT4 nibble unpack/sign-extension adds compute. This is reported as
a rejected hypothesis, not rounded into a proportional-speed claim.
