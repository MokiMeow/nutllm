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
- [ ] **Keep sensitive tensors higher precision** (embeddings, output
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

- [ ] INT8 and INT4 models load and generate **coherent** text.
- [ ] A published table with **measured** numbers:

      | format | bits/weight | model size | perplexity | tokens/sec |
      |---|---|---|---|---|

- [ ] Decode tokens/sec improves roughly in proportion to the reduction in
      weight bytes (the bandwidth hypothesis, confirmed by measurement).
- [x] Round-trip and differential tests pass.
- [x] `make all` warning-free; `make test` green.

Implementation status: the quantized storage, kernels, derived-bound tests, and
synthetic perplexity harness are complete. End-to-end quantized language-model
generation and real-text speed/quality numbers remain open behind milestone
3's stock-model adapter. See [docs/08](../08-quantisation.md); no model-quality
or proportional-speed claim is made from the synthetic fixture.

## Notes

Garbage output after quantisation is almost always a **packing bug** (nibble
order, block boundary, scale indexing), not an accuracy limit — INT4 with
per-block scales should still be clearly coherent. Test the round-trip on a
known tensor before blaming precision.

**Next:** [Milestone 6 — Polish](milestone-6-polish.md).
