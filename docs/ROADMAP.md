# Roadmap

This document tracks remaining implementation work for **Fast LightOnOCR**.

The native Rust inference pipeline is working end to end, with Python packaging
and CLI-style examples in place. Remaining milestones focus on generation
hardening, processor parity, and non-CPU execution providers.

---

# ✅ Completed Baseline

The core native inference path is in place:

- Native Rust public API
- Tokenizer loading and text decoding
- Processor-driven image preprocessing and prompt expansion
- Vision encoder ONNX execution
- Token embedding ONNX execution
- Image-feature embedding merge
- Decoder ONNX execution with KV-cache management
- Greedy autoregressive generation
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Configurable `max_new_tokens`
- Streaming generation API and example
- FP16 and Q4 model preset options
- Dynamic ONNX Runtime loading support
- Python bindings with PyO3 and maturin
- Python `from_pretrained()` download helper using `huggingface_hub`
- Python wheel packaging (CPU/CUDA build profiles, bundled ORT, release workflow)
- Native CLI-style examples (`inference`, `streaming`) for local OCR

Existing validation covers:

- Configuration error handling
- Tokenizer behavior
- Processor resize and padding behavior
- Pixtral image-processor parity fixture
- Vision encoder output parity
- Embedding model output parity
- Decoder output parity

---

# 🚧 Current Milestones

## Generation Hardening

Strengthen generation behavior and parity with Hugging Face.

### Planned work

- Deterministic generation with a public seeded-RNG option
- Generation parity against Python
- Additional stopping criteria and decoding option coverage

---

## Processor Accuracy

Improve preprocessing parity with the reference implementation.

### Planned work

- Resolution handling
- Additional real-image parity tests
- Edge-case validation for image sizes and aspect ratios

---

## Performance

Optimize inference without changing model outputs.

### Completed

- ✅ Reuse KV-cache buffers across decode steps (in-place present-tensor copy)
- ✅ Optimize top-k / top-p sampling (select_nth + heap nucleus; avoid full-vocab sort)
- ✅ CPU ORT session tuning (shared builder: intra/inter threads, graph opt, `RuntimeOptions` wiring)

### Planned work

- Reduce remaining host allocations where measured
- Reuse ONNX input tensors across decode steps where practical
- Broader inference benchmarks (latency / tokens-per-second across presets)

---

## Execution Providers

Wire non-CPU ONNX Runtime execution providers through `RuntimeOptions` without
changing model contracts.

### Planned work

- CUDA EP registration and session wiring
- CoreML EP (macOS / Apple Silicon)
- DirectML EP (Windows)
- Expose EP / thread options through the Python bindings
- Packaging notes for GPU / accelerator runtimes (wheels may stay CPU-default)

---

# 📋 Planned Milestones

## Packaging And Release Prep

Prepare the crate for broader consumption.

### Planned work

- Public API review
- Documentation polish
- Release artifacts
- Published examples

---

# 🔬 Validation Strategy

Each milestone should be independently testable.

Whenever practical, correctness is verified against the official Python implementation.

Validation includes:

- ✅ Configuration error handling
- ✅ Tokenizer parity
- ✅ Image processor parity
- ✅ ONNX stage tensor value comparison
- 🚧 Prompt parity
- 🚧 Generated token parity
- 🚧 End-to-end OCR parity

Regression tests are added throughout development to ensure future changes preserve correctness.

---

# 💡 Future Enhancements

Potential future work includes:

- Batched inference
- Multi-page document support
- PDF support
- Quantized model benchmarks
- Additional LightOnOCR model variants
- WebAssembly
- Mobile deployment
