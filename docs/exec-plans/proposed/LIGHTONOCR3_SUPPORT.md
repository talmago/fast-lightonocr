# LightOnOCR-3 Implementation Plan

This document tracks execution stages for LightOnOCR-3 support. Model facts and
architecture requirements live in the [LightOnOCR-3 model spec](../../LIGHTONOCR3_MODEL_SPEC.md);
the current LightOnOCR-2 ONNX contracts are documented in the
[LightOnOCR-2 model spec](../../LIGHTONOCR2_MODEL_SPEC.md). The general project
roadmap is in [ROADMAP.md](../../ROADMAP.md).

## Acceptance matrix

| Checkpoint | Rust load | Python local load | Python Hub load | Plain OCR parity | Grounding parity |
| --- | --- | --- | --- | --- | --- |
| LightOnOCR-2-1B ONNX | Stage 1 | Stage 1 | Preserve current behavior | Stage 1 | N/A |
| LightOnOCR-3-1B ONNX | After its export | Required | Required | Required | Required |
| LightOnOCR-3-0.8B ONNX | After Qwen3.5 backend/export | Required | Required | Required | Required |
| LightOnOCR-3-4B ONNX | After Qwen3.5 backend/export | Required | Required | Required | Required |

For each supported checkpoint, verify config dispatch, tokenizer/template IDs,
preprocessed image dimensions and values, vision output, prefill logits, one or
more decode steps, final token IDs, and decoded text. End-to-end fixtures
should include a receipt, a dense or multi-column page, and (for grounding) a
chart or embedded image. Keep fixtures small and redistributable; document any
external model assets needed for integration checks.

## Stage 1 — Lock in the LightOnOCR-2 compatibility baseline

- Add a regression check that loads a known LightOnOCR-2 ONNX checkpoint
  through the existing Rust `LightOnOCR::from_pretrained` API and runs OCR on a
  fixed image. Use the official Hugging Face repository
  `onnx-community/LightOnOCR-2-1B-ONNX` and download it into
  `models/LightOnOCR-2-1B-ONNX` so the fixture is named after its source
  checkpoint and can sit alongside future LightOnOCR-3 exports.
- Add the equivalent Python wrapper check through
  `LightOnOCR.from_pretrained()` and `process()` using the existing default
  preset and default prompt behavior.
- Record expected loading behavior, prompt/tokenizer output, and OCR output (or
  stable token IDs where full generated text can vary by runtime) as the
  baseline for later stages.
- Keep public signatures, defaults, preset names, and result types unchanged.

**Exit criteria:** regression checks pass against current LightOnOCR-2 model
artifacts before model-family refactoring begins.

## Stage 2 — Complete ONNX contract discovery

- Checkpoint config, tokenizer, processor, chat-template, generation-config,
  and safetensor metadata inspection is complete for all three LightOnOCR-3
  variants. The observed facts are recorded above. Pin repository commit SHAs
  before using these assets as parity or export fixtures.
- Export and inspect ONNX inputs/outputs, dtypes, cache/state shapes and
  update behavior, image preprocessing, prompt template, special tokens, and
  generated-output conventions against the current
  [`LIGHTONOCR2_MODEL_SPEC.md`](../../LIGHTONOCR2_MODEL_SPEC.md) and [`LIGHTONOCR3_MODEL_SPEC.md`](../../LIGHTONOCR3_MODEL_SPEC.md).
- Keep graph contracts separate by architecture family and variant where they
  differ. Config values help validate the graphs but do not replace graph
  inspection.
- Treat checkpoint config as the architecture selector; distinguish
  the existing LightOnOCR-2 `lighton_ocr` family, LightOnOCR-3-1B's
  `mistral3` wrapper, and Qwen3.5's `qwen3_5` wrapper explicitly. The 3-1B
  wrapper can share the old inference path after adapting config/tokenizer
  differences; it must not be misclassified as a Qwen3.5 backend.
- Return a clear unsupported-family or invalid-config error for unknown
  values. Resolve decoder state details, including any linear-attention state,
  from exported graph signatures before designing the Qwen3.5 runtime path.

**Exit criteria:** contracts and required assets are documented for each
architecture, with parity fixtures or reference outputs identified.

## Stage 3 — Add architecture-aware Rust loading and inference

- Introduce a small internal architecture selection layer based on
  top-level `model_type` and `architectures` in `config.json`, retaining the
  current implementation as the LightOnOCR-2 backend.
- Add a LightOnOCR-3-1B compatibility adapter for its `mistral3` wrapper,
  `image_token_id` field, tokenizer specials, and generation defaults; reuse
  the LightOnOCR-2 vision/embedding/full-attention decoder path after parity
  proves the contracts match.
- Implement the Qwen3.5 backend against its exported ONNX contracts, including
  family-specific config parsing, Qwen3VL image preprocessing and
  chat-template rendering, multimodal positions, hybrid recurrent/KV state,
  decoding, and stop-token handling.
- Share only components whose contracts are demonstrably common; keep
  architecture-specific tensor transformations explicit.
- Keep the public `LightOnOCR` loader and result surface stable. Add public API
  only if a user control cannot be derived from checkpoint configuration.
- Validate config dispatch and each pipeline stage against the official
  Python/reference implementation, then run end-to-end OCR parity for all
  three LightOnOCR-3 sizes and the LightOnOCR-2 baseline.

**Exit criteria:** supported checkpoints auto-select the correct backend and
pass stage-level and end-to-end parity checks.

## Stage 4 — Extend the Python wrapper and asset downloads

- Preserve current `from_pretrained()` arguments and behavior for
  LightOnOCR-2 repositories.
- Ensure download patterns include each architecture's actual required
  configuration, tokenizer, processor, and ONNX assets; do not infer filenames
  from the existing LightOnOCR-2 presets when exports differ.
- Keep preset selection compatible. Determine whether existing `default`,
  `fp16`, and `q4` names apply per repository from available artifacts; report
  unsupported combinations clearly rather than downloading incomplete
  checkpoints.
- Confirm local-directory loading and Hugging Face Hub loading both rely on
  the same Rust checkpoint-config dispatch.
- Document transcription and grounding prompts as model behavior without
  changing the existing default OCR prompt/API.

**Exit criteria:** Python can load local and Hub checkpoints for every
supported variant with no LightOnOCR-2 caller changes.

## Stage 5 — Export, package, and release the new ONNX models

- Add reproducible Torch-to-ONNX export scripts or documented export commands
  for the Qwen3.5 variants, preserving source checkpoints and recording
  exporter/runtime versions.
- Save exports locally for development and regression runs; keep large model
  artifacts out of source control unless repository policy establishes an
  artifact store.
- Validate exported graph parity against official Transformers outputs,
  including higher-resolution inputs and grounding/chart examples.
- Publish or consume versioned ONNX repositories, update Python download
  patterns and examples, and benchmark latency/memory for CPU and supported
  execution providers.

**Exit criteria:** repeatable exports, documented asset provenance, parity
coverage, and working package downloads exist for all supported LightOnOCR-3
variants.
