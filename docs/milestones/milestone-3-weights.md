# Milestone 3 — Real weights, real text

**Goal:** load a real open-weights model from disk and generate coherent text.
This is the milestone where nutllm becomes a working LLM engine.

## Concepts

Model file formats (GGUF / safetensors), `mmap` for zero-copy loading, tensor
naming and layout conventions, and sampling.

## Tasks

- [ ] **Loader**: parse a real format — **safetensors** is the simpler start
      (JSON header + raw tensor blob); GGUF is the llama.cpp-compatible option
      and worth it if you want to reuse existing quantised files.
- [ ] **`mmap` the file** and point tensors at it rather than copying — a
      multi-GB model should not be read into a second buffer.
- [ ] **Map names to the model**: build a table from the file's tensor names
      (`model.layers.0.self_attn.q_proj.weight`, …) to the internal structures;
      fail loudly on a missing or mis-shaped tensor rather than reading garbage.
- [ ] **Config**: read layers/heads/dim/vocab from the file's metadata, not
      hard-coded constants.
- [ ] **Generation loop**: prompt → tokenize → forward → logits → sample →
      append → repeat until EOS or a length limit.
- [ ] **Sampling**: greedy (argmax) first — it is deterministic and therefore
      testable — then temperature and top-p.
- [ ] **CLI**: `nutllm --model <path> --prompt "..." --max-tokens N`.
- [ ] **Tiny synthetic checkpoint** for CI: a 2-layer, tiny-dim model written by
      a script, so the loader and generation path are tested without downloading
      gigabytes.

## Files

`src/loader.cpp`, `include/loader.hpp`, `src/generate.cpp`, `src/main.cpp`,
`scripts/make_tiny_checkpoint.py`, `docs/03-model-formats.md` (new).

## Definition of Done

- [ ] A real small open-weights model (1B–3B) loads and generates **coherent
      English** from a prompt.
- [ ] Greedy sampling is deterministic: the same prompt gives the same output
      every run.
- [ ] The tiny synthetic checkpoint loads and generates in CI without any
      download.
- [ ] A malformed/truncated file is rejected with a clear error, not a crash.
- [ ] `make all` warning-free; `make test` green.

## Notes

Incoherent output almost always means a **layout** bug, not a math bug: a
transposed weight, wrong head stride, or an off-by-one in RoPE positions. Debug
by comparing intermediate tensors against a reference dump for the same weights,
layer by layer, until they diverge — the first divergent layer is the bug.

**Next:** [Milestone 4 — KV cache](milestone-4-kv-cache.md).
