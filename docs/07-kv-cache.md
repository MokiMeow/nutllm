# 07 — KV cache

*Milestone 4.* The optimisation that makes generation practical — and the point
where the performance problem changes shape entirely.

## The problem

Generating token *n+1* needs attention over all previous tokens. Recomputing K
and V for every earlier token at every step makes generation O(n²) in work, and
you would redo identical arithmetic thousands of times.

## The fix

Cache K and V per layer, per head, and append one column per generated token:

```
prefill  (prompt, n tokens):  compute K,V for all n     -> big matmuls
decode   (each new token):    compute K,V for 1 token,
                              append to the cache,
                              attend against the whole cache
```

Cost per generated token drops from O(n²) to O(n).

## Two regimes, two bottlenecks

This is the key insight of the whole engine:

| phase | shape | bound by |
|---|---|---|
| **prefill** | matrix × matrix | **compute** — this is what milestone 0's kernels are for |
| **decode** | matrix × *vector* | **memory bandwidth** — every weight is read to produce one token |

During decode you read the entire weight matrix to multiply it by a single
vector, so arithmetic intensity collapses and the machine is limited by how fast
it can stream weights from RAM. **This is why quantisation
([docs/08](08-quantisation.md)) is the single biggest decode speedup**: INT4
weights are a quarter the bytes, so the bandwidth-bound phase goes roughly four
times faster — even though the arithmetic is unchanged.

It is also why a register-blocked GEMM kernel helps prefill enormously and
decode barely at all. Optimise each phase for its own bottleneck.

## Memory cost

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × max_seq × sizeof(elem)
```

The implementation stores `[layer][kv_head][position][head_dim]`, so one KV
head's history is contiguous during attention. It allocates the full capacity
once and never moves it during generation. For ordinary multi-head attention
`kv_heads == heads`; grouped-query attention lets several query heads share one
KV head and reduces cache memory in direct proportion.

This grows linearly with context length and is often larger than the model
itself at long context. Mitigations (stretch goals): quantise the cache to INT8,
more aggressive KV quantisation or a sliding window. Grouped-query attention
is implemented and used by the validated TinyLlama checkpoint (32 query heads,
4 KV heads).

## Implementation notes

- **Preallocate** the cache for `max_seq` at load time; growing it mid-generation
  means reallocation and copying in the hot path.
- **Layout matters**: store K as `[layer][kv_head][seq][head_dim]` so attention
  reads contiguously along the sequence axis.
- **Correctness test**: generating with the cache must produce *identical*
  tokens to generating without it (recomputing every step). Any divergence is a
  cache bug — this differential test is the milestone's Definition of Done.

## Measured result

Median of three runs on a 13th Gen Intel Core i7-13650HX under WSL2, one
thread, `-O3 -march=native`, using a deterministic synthetic one-layer model
(`dim=64`, four heads, FFN 128):

| context | prefill tokens/s | incremental decode µs/token |
|---:|---:|---:|
| 16 | 200,652 | 5.9 |
| 64 | 131,621 | 11.4 |
| 128 | 92,037 | 18.2 |

An eightfold context increase makes incremental decode about 3.1× slower,
rather than re-running an increasingly large full transformer prefix. The
remaining linear rise is the required attention scan over cached history;
per-token decode is O(context), not constant-time. Prefill and decode are
reported separately because they exercise different kernel shapes.

For this benchmark's configured capacity the cache reports 131,072 bytes,
exactly:

```
2 × 1 layer × 4 heads × 16 head_dim × 256 positions × 4 bytes
```
