#!/usr/bin/env python3
"""Write nutllm's deterministic two-layer safetensors CI fixture."""

import json
import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def zeros(rows, columns):
    return [0.0] * (rows * columns)


def add(tensors, name, shape, values):
    expected = 1
    for dimension in shape:
        expected *= dimension
    if len(values) != expected:
        raise ValueError(f"{name}: expected {expected} values")
    tensors.append((name, shape, values))


def main():
    dim = 4
    ffn = 4
    layers = 2
    vocab = 4
    tensors = []

    embeddings = zeros(vocab, dim)
    for token in range(vocab):
        embeddings[token * dim + token] = 1.0
    add(tensors, "model.embed_tokens.weight", [vocab, dim], embeddings)

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(tensors, f"{prefix}.self_attn.{projection}.weight",
                [dim, dim], zeros(dim, dim))
        for projection in ("gate_proj", "up_proj"):
            add(tensors, f"{prefix}.mlp.{projection}.weight",
                [ffn, dim], zeros(ffn, dim))
        add(tensors, f"{prefix}.mlp.down_proj.weight",
            [dim, ffn], zeros(dim, ffn))
        add(tensors, f"{prefix}.input_layernorm.weight", [dim], [1.0] * dim)
        add(tensors, f"{prefix}.post_attention_layernorm.weight",
            [dim], [1.0] * dim)

    add(tensors, "model.norm.weight", [dim], [1.0] * dim)
    lm_head = zeros(vocab, dim)
    lm_head[1 * dim + 0] = 10.0  # H -> i
    lm_head[2 * dim + 1] = 10.0  # i -> !
    lm_head[3 * dim + 2] = 10.0  # ! -> EOS (newline token)
    add(tensors, "lm_head.weight", [vocab, dim], lm_head)

    metadata = {
        "nutllm.layers": str(layers),
        "nutllm.heads": "1",
        "nutllm.dim": str(dim),
        "nutllm.head_dim": str(dim),
        "nutllm.ffn_dim": str(ffn),
        "nutllm.vocab_size": str(vocab),
        "nutllm.max_seq": "16",
        "nutllm.eos_token": "3",
    }
    header = {"__metadata__": metadata}
    payload = bytearray()
    for name, shape, values in tensors:
        begin = len(payload)
        payload.extend(struct.pack(f"<{len(values)}f", *values))
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }

    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((-(8 + len(encoded))) % 4)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    checkpoint = FIXTURES / "tiny.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (FIXTURES / "tiny.vocab").write_text(
        "0 48\n1 69\n2 21\n3 0a\n", encoding="ascii")
    (FIXTURES / "tiny.merges").write_text("", encoding="ascii")
    write_llama2_fixture()
    print(f"wrote {checkpoint} ({checkpoint.stat().st_size} bytes)")


def write_llama2_fixture():
    dim = 4
    ffn = 4
    layers = 2
    vocab = 6
    sequence = 16
    payload = bytearray(struct.pack(
        "<7i", dim, ffn, layers, 1, 1, -vocab, sequence))

    def floats(values):
        payload.extend(struct.pack(f"<{len(values)}f", *values))

    embeddings = zeros(vocab, dim)
    embeddings[3 * dim + 0] = 1.0  # H
    embeddings[4 * dim + 1] = 1.0  # i
    embeddings[5 * dim + 2] = 1.0  # !
    floats(embeddings)
    floats([1.0] * (layers * dim))  # attention norms
    for _ in range(4):              # q, k, v, o
        floats(zeros(layers * dim, dim))
    floats([1.0] * (layers * dim))  # FFN norms
    for _ in range(3):              # gate, down, up
        floats(zeros(layers * dim, ffn))
    floats([1.0] * dim)             # final norm
    floats([0.0] * (sequence * dim))  # legacy RoPE tables
    shared = bytearray(payload)
    struct.pack_into("<i", shared, 5 * 4, vocab)
    (FIXTURES / "tiny-llama2-shared.bin").write_bytes(shared)
    classifier = zeros(vocab, dim)
    classifier[4 * dim + 0] = 10.0  # H -> i
    classifier[5 * dim + 1] = 10.0  # i -> !
    classifier[1 * dim + 2] = 10.0  # ! -> BOS/EOS
    floats(classifier)
    (FIXTURES / "tiny-llama2.bin").write_bytes(payload)

    vocabulary = [b" ", b"<s>", b"</s>", b"H", b"i", b"!"]
    tokenizer = bytearray(struct.pack("<I", 4))
    for token in vocabulary:
        tokenizer.extend(struct.pack("<fi", 0.0, len(token)))
        tokenizer.extend(token)
    (FIXTURES / "tiny-llama2.tokenizer").write_bytes(tokenizer)


if __name__ == "__main__":
    main()
