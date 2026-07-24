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
bytes = 2 (K and V) × layers × heads × head_dim × max_seq × sizeof(elem)
```

This grows linearly with context length and is often larger than the model
itself at long context. Mitigations (stretch goals): quantise the cache to INT8,
grouped-query attention (share K/V across heads), or a sliding window.

## Implementation notes

- **Preallocate** the cache for `max_seq` at load time; growing it mid-generation
  means reallocation and copying in the hot path.
- **Layout matters**: store K as `[head][seq][dim]` so the attention matmul
  reads contiguously along the sequence axis.
- **Correctness test**: generating with the cache must produce *identical*
  tokens to generating without it (recomputing every step). Any divergence is a
  cache bug — this differential test is the milestone's Definition of Done.
