# 11: Loader validation and trust boundaries

## Review scope and evidence

This pass reviews the external-file boundary, not a new model format or an
inference speedup. It preserves the dependency-free C++17 implementation,
permanent scalar references, matrix layout and existing generation outputs.
The existing roadmap is complete; this is maintenance of its loading and
verification foundations.

Primary sources reviewed on 2026-11-06:

- [Safetensors format and safety notes](https://github.com/huggingface/safetensors#format):
  a bounded JSON header, duplicate-key rejection and a completely indexed,
  nonoverlapping data buffer.
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259):
  number syntax, Unicode escapes and interoperable UTF-8 strings. Parser depth
  limits are an implementation constraint, not a tensor-format feature.
- [llama2.c reference](https://github.com/karpathy/llama2.c/blob/master/run.c):
  tokenizer records contain an fp32 score, int32 byte length and token bytes;
  special token IDs include BOS 1 and EOS 2. This is a layout reference, not
  evidence that its input validation should be copied.
- [GCC instrumentation options](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html),
  [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and
  [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html):
  compile and link instrumentation, preserve frame pointers, and fail the test
  process on findings rather than continuing silently.

## Findings and decisions

1. The previous safetensors checks validated each range independently. Two
   tensors could alias the same bytes, or leave unindexed gaps and trailing
   payload. Sort descriptors by physical offset and require exact adjacency
   from byte zero through the entire payload. Cost is O(t log t) for t tensors,
   without reading or copying all weight bytes.
2. The JSON parser had unbounded recursive calls. Reject headers above
   100,000,000 bytes and values deeper than 32 nesting levels before descending.
   Ordinary tensor descriptors need only a few levels. Reject leading-zero
   integers, malformed UTF-8 and unpaired surrogates; decode valid Unicode
   escapes before duplicate-key checks.
3. Model metadata drove allocations before expected tensors were checked.
   Preflight the complete tensor set and all expected shapes, bounding the layer
   loop by the available descriptor count, then allocate and transpose. Token
   vocabularies must fit the engine's signed integer token IDs.
4. The llama2 tokenizer reserved its declared vocabulary and allocated each
   declared string before validating remaining bytes. Reuse its checkpoint
   file reader, validate the minimum record footprint before reserve, and check
   each token against remaining file bytes before allocating it. Reject
   duplicate token strings rather than silently choosing the first ID.
5. `make test` previously included benchmark workloads despite its documented
   correctness-only purpose. `--test` now runs the same reference checks without
   timing, making instrumentation practical. Automatic header dependencies
   avoid stale incremental binaries. CI covers scalar and AVX2/FMA optimized
   builds plus a separate scalar ASan/UBSan build.

## Regression coverage

`src/loader_validation_selftest.cpp` builds temporary, tiny fixtures locally and
checks specific rejection diagnostics. Cases cover truncated lengths and
headers, excessive header/depth declarations, duplicate keys (including escaped
aliases), overlapping/gapped/trailing extents, malformed integers, shape
multiplication overflow, misalignment and out-of-bounds ranges, malformed UTF-8
and Unicode escapes, very large mismatched metadata, short tokenizer records,
duplicate tokens and trailing tokenizer bytes. Valid out-of-order descriptors,
Unicode names, and existing fp32/INT8/INT4 generation remain accepted.

The large declarations are rejected without allocating the declared model or
token. The tests do not download weights, require a JSON library, or benchmark
malformed input. See [testing](09-testing-and-benchmarking.md) for the build
matrix and sanitizer command.

## Compatibility and limits

This is still an F32-only safetensors subset: nonempty tensor shapes, aligned
float payloads, little-endian host, the documented nutllm metadata and model
names. Scalar and zero-element tensors remain unsupported. Valid Unicode
escapes are now accepted; ambiguous or malformed inputs accepted previously may
now be rejected. llama2 tokenizer vocabularies require at least three entries.

Validation is not a sandbox or a signature check. Large *valid* models and
contexts can still exceed available RAM, and tensor values may contain NaNs or
infinities. Use trusted, hash-verified files and appropriate process resource
limits. Do not modify or truncate files while they are mapped: read-only mapping
does not make another process's writes safe. File mutation protection, global
memory budgets, broader format support and fuzzing campaigns are separate work.

The tiny fixtures establish regression correctness, not new TinyLlama quality,
throughput, or real-model compatibility measurements. Existing published
benchmarks retain their original machine and conditions.
