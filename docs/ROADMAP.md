# Roadmap

This roadmap highlights future work for **Fast LightOnOCR**. The current Rust
and Python inference path supports LightOnOCR-2. Model specifications are in
[LIGHTONOCR2_MODEL_SPEC.md](LIGHTONOCR2_MODEL_SPEC.md) and
[LIGHTONOCR3_MODEL_SPEC.md](LIGHTONOCR3_MODEL_SPEC.md).

## Planned work

### LightOnOCR-3 support

The proposed execution plan starts by locking in the LightOnOCR-2 compatibility
baseline, then adds config-based model-family loading and support for the new
architectures. Follow the [LightOnOCR-3 execution plan](exec-plans/proposed/LIGHTONOCR3_SUPPORT.md).

### Generation and processor parity

- Add deterministic generation controls and broaden parity coverage against
  the official implementation.
- Improve image resize/resolution parity and cover more image sizes and aspect
  ratios.

### Execution providers

- Add CoreML on macOS and DirectML on Windows.
- Consider CUDA vision/embed residency and cache binding improvements when
  measurements justify them.

### Packaging and release

- Review the public API, polish documentation, and prepare release artifacts
  and published examples.

## Completed work

The native Rust inference pipeline, Python bindings and Hub download helper,
streaming generation, FP16/Q4 presets, CPU runtime tuning, and CUDA execution
path are in place. Validation already covers configuration, tokenizer and
processor behavior, and ONNX stage outputs.

## Later candidates

Batched and multi-page inference, PDF input, broader quantized benchmarks,
WebAssembly, and mobile deployment remain possible follow-on work.
