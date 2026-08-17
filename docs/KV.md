# Decoder KV Cache (Host and CUDA)

This document describes the host and CUDA KV-cache design for the decoder.

High-level architecture still lives in [`ARCHITECTURE.md`](ARCHITECTURE.md).
ONNX tensor contracts live in [`MODEL_CONTRACTS.md`](MODEL_CONTRACTS.md).

---

## Goals

- Keep the **host** [`KVCache`](../src/model/decoder/kv_cache.rs) strategy
  correct, simple, and the default for CPU.
- Provide a **CUDA-only** device-resident KV path that does not make
  `KVCache` device-aware.
- Prefer **one generate loop** and a clear step abstraction over
  copy-pasted control flow.
- Optimize GPU decode on the CUDA backend without changing host semantics.

---

## Naming

| Name | Kind | Role |
|------|------|------|
| `KVCacheBackend` | `pub(crate)` trait | Pluggable past/present strategy used by generate / step |
| `KVCache` | public struct | **Host-resident** backend (CPU path; also public `decode`). Implements `KVCacheBackend`. |
| `CudaKVCache` | `pub(crate)` struct in `cuda_backend` (`cfg(cuda)`) | **CUDA-resident** backend for IoBinding decode |
| `ActiveKVCache` | `pub(crate)` enum | Selected backend for one generate run (`Host` / `Cuda`) |
| `CudaIoContext` | `pub(crate)` struct in `cuda_backend` (`cfg(cuda)`) | CUDA/CPU `MemoryInfo` plus reusable embeds/mask staging |

Factory:

```text
create_kv_cache_backend(ep, batch)
  Cuda EP and FAST_LIGHTONOCR_CUDA_HOST_KV unset  → ActiveKVCache::Cuda(CudaKVCache)
  otherwise                                       → ActiveKVCache::Host(KVCache)
```

---

## Design

### One `generate_streaming` loop

```text
kv = create_kv_cache_backend(ep, batch)
cuda_io = Some(CudaIoContext) if Cuda backend else None
for step in 0..max_new_tokens:
    logits = decode_step(input, mask, &mut kv, FinalPosition, cuda_io)
    token  = next_token(logits)
    embed_into(token) ; update mask
```

### One step entry, two strategies

- Shared: shape checks (host), logits extraction / final-position materialization.
- `KVCache` → `decode_step_host`: `Session::run` + present→past host buffer update.
- `CudaKVCache` → `decode_step_cuda`: IoBinding + promote device present; embeds/mask
  copied through `CudaIoContext` staging buffers (capacity retained across steps).

Public [`Decoder::decode`](../src/model/decoder/decoder.rs) always uses host `KVCache`.

### Prefill contract (`CudaKVCache`)

Empty past (`seq=0`) is allocated on the **host** on purpose: zero-length CUDA
past tensors + IoBinding have segfaulted on some stacks (e.g. Colab). After the
first step, `promote_present` keeps present outputs on device.

```text
Prefill (step 0):  host empty past ──IoBinding──► device present ──promote──► device past
Decode (step 1+):  device past     ──IoBinding──► device present ──promote──► device past
                   host embeds/mask                 host logits (sample)
```

Sampling always runs on the host from final-position logits.

---

## Escape hatches

| Env | Effect |
|-----|--------|
| `FAST_LIGHTONOCR_CUDA_HOST_KV` | Force host `KVCache` even when EP is CUDA (debug / parity) |

---

## Validation matrix

| Build | EP | KV backend | Expect |
|-------|----|------------|--------|
| no `cuda` feature | CPU | host | unchanged host path |
| `--features cuda` | CPU | host | host path |
| `--features cuda` | CUDA | `CudaKVCache` (default) | correct OCR; GPU memory in use |
| `--features cuda` + `FAST_LIGHTONOCR_CUDA_HOST_KV` | CUDA | host | correct OCR; KV traffic via host |

---

## Follow-ons (not blocking)

- Confirm vision/embed sessions stay on CUDA EP without surprise host copies.
- Device-side logits / sampling only if product needs it.
- Batch size > 1 for `CudaKVCache` if required later.
- Reuse a single IoBinding object across steps if ORT API allows safely.

---

## References

- Implementation: [`src/model/decoder/decoder.rs`](../src/model/decoder/decoder.rs),
  [`src/model/decoder/kv_cache.rs`](../src/model/decoder/kv_cache.rs),
  [`src/model/decoder/cuda_backend.rs`](../src/model/decoder/cuda_backend.rs)
  (`--features cuda` only)
- Contracts: past/present shapes in [`MODEL_CONTRACTS.md`](MODEL_CONTRACTS.md)
- Roadmap EP section: [`ROADMAP.md`](ROADMAP.md)
