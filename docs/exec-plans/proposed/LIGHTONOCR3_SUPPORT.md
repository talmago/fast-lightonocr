# LightOnOCR-3 Implementation Plan

This document tracks execution stages for LightOnOCR-3 support. Model facts and
architecture requirements live in the [LightOnOCR-3 model spec](../../LIGHTONOCR3_MODEL_SPEC.md);
the current LightOnOCR-2 ONNX contracts are documented in the
[LightOnOCR-2 model spec](../../LIGHTONOCR2_MODEL_SPEC.md). The general project
roadmap is in [ROADMAP.md](../../ROADMAP.md).

## Execution status (2026-11-20)

- Stage 1 setup is present: the LightOnOCR-2 checkpoint is downloaded under
  `models/LightOnOCR-2-1B-ONNX` (ignored by git), a fixed image fixture and
  opt-in Rust/Python wrapper checks are added, and its Hub revision is pinned.
  The known-answer output has not been captured because ONNX Runtime is not
  installed on this host; Stage 1's parity exit criterion remains open.
- Stage 2 config, tokenizer, processor, template, generation-config, and
  safetensor metadata findings are recorded with pinned source revisions in
  `LIGHTONOCR3_MODEL_SPEC.md`. The 3-1B graph contracts are now inspected;
  the 0.8B/4B Qwen3.5 graph signatures and state behavior remain open, so
  Stage 2's all-family exit criterion is still pending.
- Stage 3 export and verification is complete for LightOnOCR-3-1B. The
  reproducible exporter and pinned requirements are in `scripts/`, with local
  graphs and SHA-256 provenance in the ignored
  `models/LightOnOCR-3-1B-ONNX/` directory. ONNX Runtime CPU parity passed for
  vision features at two image sizes, token embeddings, prefill and cached
  decode logits, and eight greedy output tokens. Contract details are in
  `LIGHTONOCR3_MODEL_SPEC.md`; the next implementation stage is Stage 4.

## Acceptance matrix

| Checkpoint | Rust load | Python local load | Python Hub load | Plain OCR parity | Grounding parity |
| --- | --- | --- | --- | --- | --- |
| LightOnOCR-2-1B ONNX | Stage 1 | Stage 1 | Preserve current behavior | Stage 1 | N/A |
| LightOnOCR-3-1B ONNX | Stage 4 | Stage 6 | Stage 6 | Stage 4 | Stage 4 |
| LightOnOCR-3-0.8B ONNX | Stage 5 | Stage 6 | Stage 6 | Stage 5 | Stage 5 |
| LightOnOCR-3-4B ONNX | Stage 6 | Stage 6 | Stage 6 | Stage 6 | Stage 6 |

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

## Stage 3 — Export and verify LightOnOCR-3-1B

The checkpoint inspector and export command are architecture-generic:
`scripts/inspect_torch.py` detects supported configurations and
`scripts/export_onnx.py` exports the matching full-attention architecture.
`scripts/inspect_onnx.py` remains dedicated to inspecting exported graphs.
Each script runs independently; model version is recorded as provenance and
does not select the export path.

- Create a pinned, isolated reference environment using a Transformers version
  that supports the `mistral3` checkpoint, plus the required export/runtime
  tooling. Keep these development dependencies out of the Rust runtime.
- Add a reproducible export script or command for the 3-1B checkpoint at its
  pinned source revision. Store ONNX graphs and external data in the `onnx/`
  subdirectory of ignored local `models/LightOnOCR-3-1B-ONNX`; copy only the
  model/processor/generation configs, tokenizer config/data, and required chat
  template to the checkpoint root (excluding PyTorch safetensors).
- Inspect graph filenames, opset/IR, input and output names, ordering, dtypes,
  ranks, dynamic dimensions, and cache behavior. Confirm whether the graphs
  match LightOnOCR-2's three-session Pixtral/full-attention contract.
- Compare processor outputs, prompt/template IDs, vision output, prefill and
  decode outputs, generated IDs, and decoded OCR text against the official
  Transformers reference. Record export tool versions and SHA-256 hashes.

**Exit criteria:** reproducible local 3-1B exports exist, their graph contract
is documented, and reference parity outputs establish whether the existing
Rust inference stages can be reused.

## Stage 4 — Add LightOnOCR-3-1B Rust compatibility

- Add the smallest architecture selector/config adapter needed for the
  `mistral3` wrapper, `image_token_id`, tokenizer specials, and generation
  defaults. Keep the existing LightOnOCR-2 path unchanged.
- Reuse the current vision, embedding, and full-attention decoder path only
  for contracts confirmed by Stage 3 parity. Keep 3-1B-specific differences
  explicit in configuration and preprocessing.
- Preserve the public loader and result surface. Validate each pipeline stage
  and end-to-end OCR against the Stage 3 reference fixtures.

**Exit criteria:** Rust loads the local 3-1B export and passes stage-level and
plain OCR parity alongside the LightOnOCR-2 baseline.

## Stage 5 — Export LightOnOCR-3-0.8B and implement Qwen3.5

- Export the pinned 0.8B checkpoint with the same reproducible, isolated
  workflow. Keep ONNX data under ignored `models/LightOnOCR-3-0.8B-ONNX/`.
- Inspect and document its actual vision, embedding, prefill, decode, image
  packing, multimodal position, recurrent linear-attention state, and KV
  cache graph contracts before choosing Rust tensor/state representations.
- Compare preprocessing, prompt/template IDs, intermediate outputs, decode
  steps, token IDs, plain OCR, and grounding against the official reference.
- Implement architecture dispatch and the Qwen3.5 backend against those
  observed contracts, including hybrid recurrent/KV state and stop handling.
- Keep architecture-specific tensor transformations explicit and add shared
  components only where graph and parity evidence show matching contracts.

**Exit criteria:** Rust loads the local 0.8B export and passes stage-level,
plain OCR, and grounding parity while preserving LightOnOCR-2 and 3-1B results.

## Stage 6 — Extend model coverage and Python downloads

- Export and inspect the pinned 4B Qwen3.5 checkpoint, then validate it with
  the Stage 5 backend and add parity fixtures for plain OCR and grounding.
- Preserve current Python `from_pretrained()` arguments and LightOnOCR-2
  behavior. Derive download patterns from each export's actual assets and
  report unsupported preset/model combinations clearly.
- Confirm local and Hub loading use the same Rust config dispatch. Keep
  `default`, `fp16`, and `q4` names compatible where the checkpoint supplies
  those artifacts.
- Save development exports locally and keep large model files out of git.
  Publish or consume versioned ONNX repositories and benchmark CPU and
  supported execution providers after parity passes.

**Exit criteria:** all three LightOnOCR-3 sizes load through Rust and Python,
pass stage-level and end-to-end parity, and have documented export provenance
and working package downloads.
