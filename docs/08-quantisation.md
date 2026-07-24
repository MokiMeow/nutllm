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
    scale = max(|w|) / 7          # INT4 signed range is [-8, 7]
    q[i]  = round(w[i] / scale)   # stored in 4 bits
store: 32 × 4-bit values + one fp16 scale
```

Effective bits per weight = 4 + 16/32 = **4.5**, and accuracy stays close to
fp32 because outliers only affect their own block.

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
