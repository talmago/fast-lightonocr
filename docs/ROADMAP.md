# Roadmap

This document tracks the remaining implementation work for **Fast LightOnOCR**.

The native Rust inference pipeline is already working end to end. Remaining
milestones focus on generation hardening, parity coverage, packaging, and
deployment surfaces.

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

### Planned work

- Reduce allocations
- Reuse ONNX tensors
- Optimize KV-cache updates
- Benchmark inference

---

# 📋 Planned Milestones

## Python Packaging Hardening

Broaden and polish the Python distribution.

### Deliverables

- Python examples
- API parity for generation options
- Wheel metadata review
- Release workflow

---

## Command-Line Interface

Provide a native CLI.

### Planned work

- OCR from images
- Local model loading
- Generation options
- Markdown output

---

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

- GPU execution providers
- Batched inference
- Multi-page document support
- PDF support
- Quantized model benchmarks
- Additional LightOnOCR model variants
- WebAssembly
- Mobile deployment
