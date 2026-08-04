# Decoder Loop Performance

Investigation date: 2026-08-04  
Follow-on to [`KV_CACHE_PERF.md`](KV_CACHE_PERF.md).

Workload: `models/lightonocr` (default), `examples/SROIE-receipt.jpeg`,  
`LIGHTONOCR_KV_STRATEGY=reusable`, 256 new tokens, macOS CPU, ONNX Runtime 1.28.

## Goal

Attribute exclusive time inside the autoregressive generation loop so the next optimization backlog is evidence-based.

## Method

Extended `profiling` feature stages:

- decoder: tensor prep, ONNX, logits extract, KV update
- token selection: greedy argmax vs sample candidate build / top-k / top-p / draw
- decoder-loop attribution block (exclusive shares of generation)

Runner:

```bash
ORT_DYLIB_PATH=... LIGHTONOCR_PROFILE=1 LIGHTONOCR_KV_STRATEGY=reusable \
  cargo run --release --example profile_decoder_loop --features 'load-dynamic,profiling' -- \
  models/lightonocr examples/SROIE-receipt.jpeg 256 sample   # or greedy
```

## Measured attribution (256 tokens, reusable KV)

### Default config (`do_sample=true`, `temperature=0.2`, `top_k=0`, `top_p=0.9`)

| Exclusive stage | Time | Share of generation |
| --- | ---: | ---: |
| Decoder ONNX | 4596 ms | **77.7%** |
| Token selection | 855 ms | **14.4%** |
| └ sample top-p | 799 ms | **13.5%** |
| └ sample candidate build | 50 ms | 0.8% |
| └ sample top-k / draw | ~0 ms | ~0% |
| Decoder host (prep+logits+KV) | 434 ms | 7.3% |
| └ KV update | 414 ms | ~7.0% |
| └ logits extract | 17 ms | 0.3% |
| └ tensor prep | 3 ms | ~0% |
| Token embedding | 4 ms | 0.1% |
| Attention mask update | ~0 ms | ~0% |
| Generation total | 5918 ms | 100% |

### Greedy (`do_sample=false`)

| Exclusive stage | Time | Share of generation |
| --- | ---: | ---: |
| Decoder ONNX | 4607 ms | **89.8%** |
| Decoder host (prep+logits+KV) | 449 ms | 8.7% |
| Token selection (argmax) | 51 ms | 1.0% |
| Token embedding | 4 ms | 0.1% |
| Generation total | 5128 ms | 100% |

Default sampling costs ~790 ms extra vs greedy on this workload (~13% of generation), almost entirely in **top-p**.

## Root cause of expensive token selection

`generation_config.json` sets `do_sample: true` with `top_k: 0` and `top_p: 0.9`.

Current path each step:

1. Build a `Vec` of ~151936 temperature-scaled candidates.
2. `top_k=0` is a no-op.
3. `apply_top_p` **sorts the full vocabulary**, then softmax-scans until cumulative mass ≥ 0.9.

That full-vocab sort dominates host time outside ONNX/KV.

Greedy argmax over the same vocab is only ~0.2 ms/token and is not a priority.

## Ranked next optimization items

Priority is expected E2E impact for the default (`do_sample=true`) path, assuming reusable KV is already adopted.

| Priority | Item | Why | Expected impact | Effort |
| ---: | --- | --- | --- | --- |
| 1 | **Make `ReusableBuffers` the production KV default** | Proven in KV investigation; ~11% E2E at 256 tokens vs full-copy | High | Low |
| 2 | **Rewrite top-p / sampling** | 13–14% of generation today; full-vocab sort every step | High on default config | Medium |
| 2a | Early top-k / partial selection before top-p | Avoid sorting 151k when only a nucleus is needed | High | Medium |
| 2b | Reuse candidate buffers across steps | Cuts alloc churn in candidate build | Low–medium | Low |
| 3 | **ORT / execution-provider tuning** | Still ~78–90% of generation after host fixes | Highest ceiling | Medium–high |
| 4 | Extract/copy only last logit position when `seq_len==1` | Logits copy is small now (~17 ms / 256) but wasteful | Low | Low |
| 5 | Session IO binding / fixed-shape padded cache | Reduce ORT input binding overhead; may help ORT graphs | Uncertain | High |
| 6 | Fuse or batch single-token embedding session | Currently ~0.1% of generation | Very low | Low |
| — | Attention mask updates | Already negligible | None | — |
| — | Naive append-only KV | Rejected; slower than reusable | Negative | — |

### Sampling rewrite notes (item 2)

Concrete directions that preserve greedy/sample semantics:

- Prefer a bounded selection algorithm (heap / partial sort / online nucleus) instead of sorting the full vocab when `top_k==0`.
- If product defaults can change, setting a modest `top_k` (e.g. 50–100) before top-p would also collapse cost immediately with the current code.
- Keep a fast path for `do_sample=false` (already cheap).

### ORT notes (item 3)

After host KV + sampling are fixed, remaining generation time is almost all `session.run`. Next levers are outside Rust tensor bookkeeping:

- CPU ORT graph/session options (threads, arena, optimization level)
- CUDA / CoreML / other EPs where available
- Longer-term model-export changes (not in scope for host-only work)

## Reproduction checklist

```bash
# Default sampling profile
LIGHTONOCR_PROFILE=1 LIGHTONOCR_KV_STRATEGY=reusable \
  cargo run --release --example profile_decoder_loop --features 'load-dynamic,profiling' -- \
  models/lightonocr examples/SROIE-receipt.jpeg 256 sample

# Greedy baseline for host-vs-ORT contrast
LIGHTONOCR_PROFILE=1 LIGHTONOCR_KV_STRATEGY=reusable \
  cargo run --release --example profile_decoder_loop --features 'load-dynamic,profiling' -- \
  models/lightonocr examples/SROIE-receipt.jpeg 256 greedy
```

## Conclusion

With reusable KV in place, the decoder loop’s remaining host hotspot on the **default** config is **top-p sampling** (~14% of generation). After that, further gains require **ONNX Runtime / EP** work, not more Rust KV cleverness.

Recommended sequencing for next implementation PRs:

1. Flip production KV update to `ReusableBuffers`.
2. Optimize sampling/top-p (algorithm + optional buffer reuse).
3. Measure again; then pursue ORT/EP tuning against the new baseline.
