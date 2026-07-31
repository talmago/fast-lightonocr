# Roadmap

This document tracks the remaining implementation work for **Fast LightOnOCR**.

The native Rust inference pipeline is already working end to end. Remaining
milestones focus on generation options, parity hardening, packaging, and
deployment surfaces.

---

# 🚧 Current Milestones

## Sampling

Implement generation strategies equivalent to Hugging Face.

### Planned work

- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Configurable generation options
- Deterministic generation with seeded RNG
- Generation parity against Python

---

## Processor Accuracy

Improve preprocessing parity with the reference implementation.

### Planned work

- Resize validation
- Padding validation
- Resolution handling
- Additional image parity tests

---

## Performance

Optimize inference without changing model outputs.

### Planned work

- Reduce allocations
- Reuse ONNX tensors
- Optimize KV-cache updates
- Benchmark inference

---

# 📋 Planned Milestones

## Python Bindings

Expose the Rust engine through Python.

### Deliverables

- Python package
- Python examples
- API parity

---

## Command-Line Interface

Provide a native CLI.

### Planned work

- OCR from images
- Local model loading
- Generation options
- Markdown output

---

# 🔬 Validation Strategy

Each milestone should be independently testable.

Whenever practical, correctness is verified against the official Python implementation.

Validation includes:

- Configuration parity
- Tokenizer parity
- Prompt parity
- Image processor parity
- Tensor value comparison
- Generated token parity
- End-to-end OCR parity

Regression tests are added throughout development to ensure future changes preserve correctness.

---

# 💡 Future Enhancements

Potential future work includes:

- GPU execution providers
- Batched inference
- Multi-page document support
- PDF support
- Quantized model benchmarks
- Additional LightOnOCR model variants
- WebAssembly
- Mobile deployment
