# KV-Cache Performance Investigation

Investigation date: 2026-08-04  
Workload: `models/lightonocr` (default), `examples/SROIE-receipt.jpeg`, greedy decoding, macOS CPU, ONNX Runtime 1.28.

## Summary

Host-side KV-cache handling is a real bottleneck for longer generations. At 256 new tokens it accounted for about **21% of end-to-end time** under the production full-copy path.

**Recommended production design: `ReusableBuffers`** — reuse per-layer `Vec<f32>` capacity and overwrite with each full `present.*` tensor. It preserved greedy token parity and improved E2E median latency by about **11.5%** at 256 tokens.

Do **not** ship naive append-only `(B,H,S,D)` updates or alternate-layout append+transpose as the default.

## Baseline (full-copy replace)

Current production behavior copies every `present.{layer}.{key,value}` tensor into a new `Vec` each decoder step (`data.to_vec()`), then drops the previous cache.

| Metric (max_new_tokens=256, streaming profile) | Value |
| --- | ---: |
| Total inference | 8907 ms |
| ONNX Runtime total | 6548 ms (73.5%) |
| KV update/extraction | 1381 ms |
| KV extract bytes copied | 24.9 GiB |
| Generation total | 7169 ms |

At 64 tokens, KV update+replace was ~10% of total; at 256 tokens it grew to ~21% because copied bytes scale roughly with \(\sum_s s\) over decode steps.

## Why append-only regressed / underperformed

Measured in isolation (`examples/kv_cache_bench`, max_seq=256):

| Strategy | Median host time | vs full copy | Notes |
| --- | ---: | ---: | --- |
| FullCopyReplace | 322 ms | 1.00x | baseline |
| ReusableBuffers | 93 ms | 3.46x | best practical full-present path |
| DoubleBuffer | 96 ms | 3.36x | similar to reusable |
| AppendInPlaceBHSD | 166 ms | 1.94x | still rewrites full `(B,H,S,D)` |
| AppendAltLayout | 1433 ms | 0.22x | append cheap, transpose kills it |
| DeltaExtractOnly | 8 ms | 42x | microbench only (no ORT) |
| PaddedMaxLen | 92 ms | 3.50x | similar to reusable on host |

Root causes for append-only disappointment end-to-end:

1. **Layout**: ONNX requires contiguous `(batch, kv_heads, seq, head_dim)`. Appending one token is not a contiguous `Vec::extend`; each head slab must grow, forcing an O(S) rewrite.
2. **ORT still emits full `present.*`**: Reducing host “logical” append work does not shrink ORT output tensors. Delta extract still pays to read/expand around those outputs.
3. **Transpose tax**: Storing `(B,H,D,S)` for cheap append and converting to `(B,H,S,D)` each step was ~5x slower than full copy in the microbench.
4. **Allocator vs memcpy**: One contiguous memcpy into a reused buffer beats many strided writes even when the append path reports fewer “new” bytes.

E2E profile confirms this: delta cut recorded KV extract bytes from 24.9 GiB to 125 MiB, but KV stage time only fell to 933 ms — still worse than reusable’s 443 ms — because expanding `(B,H,S,D)` dominates.

## End-to-end A/B (greedy token parity)

`examples/kv_cache_e2e` (median of 3 timed `process_file` runs after warmup):

| Strategy | 64 tokens | 256 tokens | Token parity |
| --- | ---: | ---: | --- |
| full_copy | 3807 ms (1.00x) | 7910 ms (1.00x) | baseline |
| reusable | 3761 ms (1.01x) | 7094 ms (**1.12x**) | match |
| delta | 3818 ms (1.00x) | 7683 ms (1.03x) | match |

Profiled generation at 256 tokens:

| Strategy | Generation | KV update | KV bytes |
| --- | ---: | ---: | ---: |
| full_copy | 7169 ms | 1381 ms | 24.9 GiB |
| reusable | 6109 ms | 443 ms | 24.9 GiB |
| delta | 6685 ms | 933 ms | 0.12 GiB |

Reusable wins because it removes allocate/free churn and keep the hot path as contiguous memcpy, which is what this CPU prefers.

## Recommendation

1. **Adopt `ReusableBuffers` as the production KV update path** (already implemented behind `LIGHTONOCR_KV_STRATEGY=reusable`). Default remains `full_copy` until a follow-up flips the default after any extra soak testing desired.
2. Keep delta extract as an experiment only; it is not the best E2E design with the current ONNX contract/layout.
3. Reject append-only `(B,H,S,D)` and append+transpose layouts for this model family.
4. Secondary finding: default `do_sample` top-p path is expensive (~0.9 s / 256 tokens). Follow-up decoder-loop attribution is in [`DECODER_LOOP_PERF.md`](DECODER_LOOP_PERF.md).

## How to reproduce

```bash
# Stage timers + KV byte counters (compile-time feature + runtime env)
ORT_DYLIB_PATH=... LIGHTONOCR_PROFILE=1 \
  cargo run --release --example streaming --features 'load-dynamic,profiling' -- \
  models/lightonocr examples/SROIE-receipt.jpeg default 256

# Host-only representation microbench
cargo run --release --example kv_cache_bench -- 256

# E2E strategy A/B (greedy)
ORT_DYLIB_PATH=... \
  cargo run --release --example kv_cache_e2e --features 'load-dynamic,profiling' -- \
  models/lightonocr examples/SROIE-receipt.jpeg 256

# Force a strategy during any inference run
LIGHTONOCR_KV_STRATEGY=reusable|delta|full_copy
```

## Implementation notes

- Profiling is behind Cargo feature `profiling`; runtime report requires `LIGHTONOCR_PROFILE=1`.
- Experimental strategies live in `KvUpdateStrategy` / `update_cache_from_outputs` in `src/model/decoder/decoder.rs`.
- Public `Decoder::decode` still uses full-copy semantics for API compatibility with existing tests.
