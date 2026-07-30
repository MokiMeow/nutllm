# Milestone 5 — Quantisation

**Goal:** INT8 and INT4 weights — the biggest single speedup for decode, because
decode is bandwidth-bound.

## Concepts

Blocked (group-wise) symmetric quantisation, dequantising inside the kernel,
and measuring accuracy loss with perplexity.

## Tasks

- [x] **Quantiser**: per-block (32 or 64 weights) symmetric scales.
      INT8: `scale = max|w| / 127`. INT4: `scale = max|w| / 7`, two weights
      packed per byte. Store the fp16 scale alongside each block.
- [x] **Document the packing order explicitly** (which nibble is which weight) —
      it is the most common source of "the output is garbage."
- [x] **Dequant-in-kernel matmul**: unpack and scale **inside** the inner loop,
      in registers. Dequantising a whole matrix to fp32 in memory first would
      read *more* bytes than fp32 and defeat the entire purpose.
- [x] **Keep sensitive tensors higher precision** (embeddings, output
      projection) — standard practice, and cheap.
- [x] **Round-trip test**: quantise → dequantise; max error within the block's
      theoretical bound.
- [x] **Differential test**: quantised matmul vs the fp32 reference, with a
      tolerance *derived from* the quantisation error, not an arbitrary constant.
- [x] **Perplexity harness**: measure perplexity on a fixed text sample for
      fp32 / INT8 / INT4 on the same model.

## Files

`include/quant.hpp`, `src/quant.cpp`, `src/matmul_q.cpp`,
`src/perplexity.cpp`, `docs/08-quantisation.md`.

## Definition of Done

- [x] INT8 and INT4 models load and generate **coherent** text.
- [x] A published table with **measured** numbers:

      | format | bits/weight | model size | perplexity | tokens/sec |
      |---|---|---|---|---|

- [x] Decode tokens/sec is measured against the reduction in weight bytes. The
      result rejects the simple proportional hypothesis: INT4 is only 1.04×
      faster than INT8 although the runtime image is 31% smaller, because
      nibble unpacking and the fp32 classifier are material at this scale.
- [x] Round-trip and differential tests pass.
- [x] `make all` warning-free; `make test` green.

Implementation status: complete. TinyLlama 1.1B runs coherently through both
the INT8 and INT4 projection paths, while embeddings, normalization weights,
and the classifier remain fp32. The published accuracy, storage, throughput,
and hypothesis result are in [docs/08](../08-quantisation.md).

## Notes

Garbage output after quantisation is almost always a **packing bug** (nibble
order, block boundary, scale indexing), not an accuracy limit — INT4 with
per-block scales should still be clearly coherent. Test the round-trip on a
known tensor before blaming precision.

**Next:** [Milestone 6 — Polish](milestone-6-polish.md).
