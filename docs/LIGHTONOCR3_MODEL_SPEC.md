# LightOnOCR-3 Model Spec

This document records confirmed checkpoint behavior and the model-family
contracts needed for native Rust and Python support. It complements the
[LightOnOCR-2 model spec](LIGHTONOCR2_MODEL_SPEC.md). The staged execution
work is tracked in the [LightOnOCR-3 implementation plan](exec-plans/proposed/LIGHTONOCR3_SUPPORT.md).

> **Contract status:** checkpoint configs, processor/tokenizer assets, chat
> templates, generation configs, and safetensor metadata have been inspected
> and their source revisions are pinned below. A local LightOnOCR-3-1B ONNX
> export has also been inspected and checked against Transformers. Qwen3.5
> 0.8B graphs have been exported and checked against Transformers for vision,
> embeddings, prefill logits, and one cached decode step. Dynamic vision-grid
> generalization, end-to-end OCR/grounding parity, and Rust execution remain
> open; 4B contracts remain open until its export is produced.

## Pinned checkpoint revisions

These revisions identify the source checkpoint snapshots inspected for Stage 2.
They are provenance references, not ONNX export or parity fixtures.

| Repository | Revision | Inspection scope |
| --- | --- | --- |
| `lightonai/LightOnOCR-3-1B` | `23d544085c886bbdace647bcfab91203299417e2` | Config, tokenizer, processor, template, generation config, safetensor metadata |
| `lightonai/LightOnOCR-3-0.8B` | `36d8636925327c671bd8e479c517a004f7566bd9` | Config, tokenizer, processor, template, generation config, safetensor metadata |
| `lightonai/LightOnOCR-3-4B` | `b71010f095dc3735de7ee6b9969bff8f8f50b39a` | Config, tokenizer, processor, template, generation config, safetensor metadata |
| `onnx-community/LightOnOCR-2-1B-ONNX` | `2e2ecebebd69beaa3c38822c2927611eba811bd2` | Stage 1 ONNX compatibility baseline assets |

The local LightOnOCR-2 baseline model is stored under the ignored
`models/LightOnOCR-2-1B-ONNX/` directory. Its fixed image fixture is
`tests/fixtures/lightonocr2_baseline.png`; opt-in Rust and Python wrapper
smoke checks are available through `LIGHTONOCR_RUN_CHECKPOINT_REGRESSION=1
cargo test --features load-dynamic --test inference` and
`scripts/run_example.py` respectively. Both use the default
preset and default prompt behavior. The known-answer OCR text/token IDs still
need to be captured on a host with ONNX Runtime installed. Model files remain
outside source control.

## Scope and model families

The loader should select the architecture from the model checkpoint's
`config.json`, without requiring callers to choose a backend. The released
family has three sizes: LightOnOCR-3-1B retains the LightOnOCR-2 architecture,
while LightOnOCR-3-0.8B and LightOnOCR-3-4B use the Qwen3.5 vision-language
architecture. Qwen3.5's processor and runtime contracts must be established
from its official exports; it should not be assumed to fit the existing
LightOnOCR-2 graph or KV-cache path.

References: [LightOnOCR-3-1B model card](https://huggingface.co/lightonai/LightOnOCR-3-1B),
[0.8B config](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/main/config.json),
[4B config](https://huggingface.co/lightonai/LightOnOCR-3-4B/blob/main/config.json).

## Confirmed model behavior

| Model | Architecture family | Plain OCR | Grounding | Reference page size |
| --- | --- | --- | --- | ---: |
| LightOnOCR-2-1B | Existing LightOnOCR/Pixtral path | Yes | No | 1,540 px |
| LightOnOCR-3-1B | Mistral3 wrapper with Qwen3 text + Pixtral vision | Yes | Yes | 1,540 px |
| LightOnOCR-3-0.8B | Qwen3.5 VLM | Yes | Yes | 2,048 px; up to 5 MP |
| LightOnOCR-3-4B | Qwen3.5 VLM | Yes | Yes | 2,048 px; up to 5 MP |

The model's normal prompt is image-only (empty text prompt). Grounding mode is
selected by adding the literal text `grounding` alongside the page image. Its
output remains text: each layout block is prefixed with a Markdown-like marker
of the form `![label](x1,y1,x2,y2)`, where coordinates are normalized to
0–1000. Grounded image blocks include short descriptions; chart blocks contain
HTML tables. Labels include text, title, list, header, footer, page_number,
footnote, caption, formula, code, table, image, chart, header_image,
footer_image, and aside_text. A `+` suffix denotes a continuation block.
Grounding adds roughly 25% more output tokens on the cited 4B evaluation
average, so generation limits must allow for longer outputs. Plain OCR must
remain the default.

The current Python `system_prompt` is rendered as a system message, while the
official LightOnOCR-3 reference sends `grounding` as text in the user message
with the image. Preserve current prompt behavior for LightOnOCR-2. For v3,
verify exact chat-template placement and token IDs for both modes before
reusing that parameter. If the semantics differ, add an optional mode/prompt
argument with the existing default unchanged; do not silently route
`grounding` through the system-message path.

These modes describe model prompting/output, not a new structured Rust result
contract. The first compatible implementation can return the raw generated
text through the current `OCRResult`; parsing grounded blocks into typed
regions is a separate optional API decision and must not change `text()` or
the existing Python `OCRResult.text` behavior.

The Python `parse_document()` currently recognizes embedded HTML tables and
preserves everything else as Markdown text. That is sufficient to retain raw
grounding markers and chart tables, but it does not expose labels or bounding
boxes as structured fields. Add typed region parsing only as an additive
follow-up with fixtures for continuation blocks and coordinate validation.

Sources: official [0.8B model card and prompt/output description](https://huggingface.co/lightonai/LightOnOCR-3-0.8B),
[LightOnOCR-3 announcement](https://huggingface.co/blog/lightonai/lightonocr-3),
and [LightOnOCR reference client model table](https://github.com/lightonai/LightOnOCR#models).

## Checkpoint and preprocessing details

The current Qwen3.5 checkpoint configs establish several important differences
from the repository's existing contracts:

| Property | Current LightOnOCR-2 runtime | LightOnOCR-3-1B config | Qwen3.5 0.8B config | Qwen3.5 4B config |
| --- | --- | --- | --- | --- |
| Top-level `model_type` | `lighton_ocr` | `mistral3` | `qwen3_5` | `qwen3_5` |
| Text `model_type` | `qwen3` | `qwen3` | `qwen3_5_text` | `qwen3_5_text` |
| Hidden size | 1,024 | 1,024 | 1,024 | 2,560 |
| Decoder layers | 28 | 28 | 24 | 32 |
| Vocabulary size | 151,936 | 151,936 | 248,320 | 248,320 |
| Attention layer types | Full attention | Full attention | Repeating linear/full attention pattern | Repeating linear/full attention pattern |
| Declared weight dtype | float32 in current exports | bfloat16 | bfloat16 | bfloat16 |
| Vision patch / merge | Existing config-driven Pixtral processing | 14 / 2 | 16 / 2 | 16 / 2 |

The downloaded LightOnOCR-3-1B checkpoint confirms the architecture-family
statement needs precision: the wrapper is `mistral3`, its text model is
`qwen3`, and its vision model is `pixtral`. It has 28 full-attention text
layers, 1,024 hidden size, vocabulary 151,936, 8 KV heads, head dimension 128,
28 vision layers, 14-pixel patches, and spatial merge 2. It uses bfloat16
weights. This is structurally compatible with the existing LightOnOCR-2
inference stages, subject to config/tokenizer/processor validation.

There is one concrete config incompatibility to account for: 3-1B's top-level
`config.json` uses `model_type: "mistral3"`, architecture
`LightOnOCRForConditionalGeneration`, and `image_token_id`; the current Rust
decoder envelope requires `image_token_index`. Config dispatch must recognize
the actual checkpoint identifiers and normalize the image token field into
the internal family config, while preserving LightOnOCR-2 parsing. Its
processor config supplies `<|image_pad|>`, `<|vision_pad|>`, `<|vision_end|>`,
patch size 14, merge size 2, RGB normalization and a 1,540-pixel longest edge.
Its tokenizer also contains object-reference, bounding-box, quadrilateral,
vision, and video special tokens; record and test the IDs used by the
checkpoint rather than hardcoding the LightOnOCR-2 token inventory.

The 0.8B and 4B checkpoint configs mix top-level multimodal fields with nested
`text_config` and `vision_config`. At minimum, the Qwen3.5 family introduces
top-level `image_token_id`, `vision_start_token_id`, `vision_end_token_id`,
`video_token_id`, `eos_token_id`, and `pad_token_id`; text config uses
`vocab_size`, `num_hidden_layers`, `layer_types`, `linear_*` dimensions,
`rope_parameters` with multimodal RoPE fields, and `use_cache`. The 0.8B config
declares 24 layers with 18 linear-attention and 6 full-attention layers, hidden
size 1,024; 4B declares 32 layers and hidden size 2,560. The vision configs
declare temporal patch size 2 and output hidden sizes 1,024 (0.8B) and 2,560
(4B). These config values are metadata only; the ONNX graph signature remains
authoritative for runtime tensor contracts.

The Qwen3.5 processor assets identify `Qwen3VLProcessor` and
`Qwen2VLImageProcessor`, with patch size 16, merge size 2, temporal patch size
2, RGB conversion, resize, rescale factor 1/255, and per-channel mean/std of
0.5. Do not interpret the very large `size.longest_edge` values in processor
JSON as the deployment target: the official model settings use 2,048 px / 5 MP
for 0.8B and 4B. Conversely, the 1B reference target stays at 1,540 px. The
implementation must represent pixel and pixel-area limits separately if the
reference preprocessing requires both.

The current Rust processor is specifically Pixtral-shaped: it creates a
rectangular NCHW tensor, uses row-break/end markers, and expands a single image
placeholder into a row-wise sequence of image tokens. Qwen3.5 likely requires
different image packing and image/video token conventions. Implement its
preprocessor from reference inputs/outputs rather than adapting the current
row-wise expansion by assumption. Video support is outside this OCR scope.

The downloaded Qwen3.5 `chat_template.jinja` starts image content with
`<|vision_start|><|image_pad|><|vision_end|>` and inserts an empty think block
in the generation prompt when thinking is disabled. Its model card says
thinking is disabled for OCR. The 3-1B template instead uses the LightOnOCR
chat format and emits `<|image_pad|>` before processor expansion. Template
rendering must therefore be architecture-specific and compared byte-for-byte
and token-for-token with the Transformers processor; the current hand-built
template is not a shared source of truth.

Safetensor inspection confirms the structural split:

- 3-1B weights contain `model.vision_encoder.*` and
  `language_model.model.layers.*` Qwen3 full self-attention tensors.
- 0.8B weights contain `model.visual.*` and `model.language_model.layers.*`;
  linear layers include `linear_attn.A_log`, `dt_bias`, `conv1d`,
  `in_proj_qkv`, `in_proj_a/b/z`, and `out_proj`, while full-attention layers
  use `self_attn.{q,k,v,o}_proj` and q/k normalization.
- The 0.8B text stack alternates three linear-attention layers then one
  full-attention layer for 24 layers (18 linear, 6 full). Its hidden size is
  1,024, attention `head_dim` is 256, and linear key/value head dimensions are
  128. Do not derive cache dimensions from hidden size alone.
- The 4B checkpoint uses 32 text layers, 16 full-attention heads, 4 KV heads,
  hidden size 2,560, and 32 linear value heads of dimension 128. It has 24
  linear-attention and 8 full-attention layers. Its vision output dimension is
  2,560. Safetensors show the linear QKV projection is `[8192, 2560]`, the
  linear convolution is `[8192, 1, 4]`, full-attention query projection is
  `[8192, 2560]`, and K/V projections are `[1024, 2560]`. The vision patch
  projection is `[1024, 3, 2, 16, 16]`.
- 0.8B safetensors show a `[248320, 1024]` embedding matrix, a linear QKV
  projection `[6144, 1024]`, and vision patch projection
  `[768, 3, 2, 16, 16]`. 3-1B has `[151936, 1024]` embeddings, Qwen3 full
  attention projections, and Pixtral vision weights. These are architecture
  inspection findings, not ONNX contracts.

Safetensor inspection establishes weights and module structure but not runtime
state signatures. It does not replace inspection of cache/state graph inputs
and outputs, position IDs, or image patch tensors in ONNX.

Generation configs differ too. The downloaded 3-1B config provides `do_sample`
true, temperature 0.2, top-p 0.9, top-k 0, and EOS IDs 151645 and 151643. The
0.8B and 4B configs are sparse: they provide EOS 248046, pad 248044, and
`use_cache` false for 0.8B / true for 4B, but omit sampling fields and BOS.
Do not deserialize these files directly into the current required-field
`GenerationConfig`. Define family-aware defaults from reference generation
behavior, preserve user overrides, and test each stop ID. The Transformers
examples use `max_new_tokens=1024`, temperature 0.2, and thinking disabled;
validate these reference settings rather than assuming universal defaults.

## LightOnOCR-3-0.8B ONNX export contract

The pinned 0.8B snapshot was exported by the Qwen3.5 branch of
[`scripts/export_onnx.py`](../scripts/export_onnx.py), using the isolated
environment in [`scripts/requirements-lightonocr-export.txt`](../scripts/requirements-lightonocr-export.txt).
The local export and `export_report.json` are under the ignored
`models/LightOnOCR-3-0.8B-ONNX/` directory. Export uses Transformers 5.5.4,
PyTorch 2.6.0, ONNX 1.17.0, ONNX Runtime 1.32.0 CPU, opset 20, and float32
graph weights loaded from BF16 checkpoint weights. The exported checkpoint
revision is `36d8636925327c671bd8e479c517a004f7566bd9`.

The fixed SROIE fixture produced `pixel_values` `[784,1536]` (flattened
temporal/channel/patch values), `image_grid_thw` `[1,3]` with value `[1,28,28]`,
210 prompt tokens, and 196 projected image features. The 196 image placeholder
tokens match the Qwen vision merger output. `mm_token_type_ids` accompanies
the text input and is used by Transformers to create three-axis multimodal
position IDs. The Qwen template emits a vision-start/image/vision-end sequence;
it does not use the Pixtral row-break token expansion.

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `vision_encoder.onnx` | `pixel_values` float32 `[num_patches,1536]` | `image_features` float32 `[num_image_features,1024]` |
| `embed_tokens.onnx` | `input_ids` int64 `[batch,sequence]` | `inputs_embeds` float32 `[batch,sequence,1024]` |
| `decoder_prefill.onnx` | `inputs_embeds` float32 `[batch,sequence,1024]`, `position_ids` int64 `[3,batch,sequence]`, `attention_mask` int64 `[batch,total_sequence]` | `logits` float32 `[batch,sequence,248320]`; 24 layer states |
| `decoder_decode.onnx` | Same three inputs plus per-layer state tensors | `logits` and updated per-layer state tensors |

Each decoder graph stores all external tensors—including the recurrent-state
constants attached to ONNX `Constant` nodes—in its single `.onnx.data` file.
The exporter compacts both initializer data and attribute-tensor sidecars after
tracing, so a successful export leaves no per-constant sidecar files.

Layers 0–2, 4–6, 8–10, 12–14, 16–18, and 20–22 each expose
`conv_state` float32 `[batch,6144,4]` and `recurrent_state` float32
`[batch,16,128,128]`. Layers 3, 7, 11, 15, 19, and 23 expose full-attention
`key` and `value` tensors float32 `[batch,2,sequence,256]`. The prefill graph
returns those same state types; decode consumes and returns all 48 state
tensors. Position IDs are explicit graph inputs, with three axes (text/time,
height, width); Rust must calculate their multimodal layout and advance them
using the prompt's position delta during generation.

The vision exporter replaces Transformers' Python-list grid helpers with
tensor-based position interpolation and single-image packed attention. The
resulting graph retains `image_grid_thw`. A second processor-produced 700×280
image produced 792 patch rows with grid `[1,18,44]` and 198 image features;
vision outputs matched Transformers at rtol/atol 2e-3. This establishes
dynamic grid parity for the single-image OCR path exercised here, not for
multiple images or videos.

ONNX Runtime CPU matched Transformers for vision features at two image grids
(rtol/atol 2e-3), embeddings (1e-5), prefill logits, and one cached decode
step (3e-3).
These stage checks do not yet establish generated-token, plain OCR, or
grounding parity. Graph hashes and complete tensor signatures are in the
local export report.

LightOnOCR-3-1B can reuse the existing full-attention inference stages only
after parity confirms matching tensor contracts. Its model wrapper/config
adapter remains necessary, and its ONNX filenames, numeric precision, and
preprocessing must be read from its own export rather than inferred from
LightOnOCR-2 architecture continuity.

## LightOnOCR-3-1B ONNX export contract

The pinned 3-1B checkpoint was exported locally with the standalone,
architecture-generic [`scripts/export_onnx.py`](../scripts/export_onnx.py).
Checkpoint metadata can be inspected independently with
[`scripts/inspect_torch.py`](../scripts/inspect_torch.py), while
[`scripts/inspect_onnx.py`](../scripts/inspect_onnx.py) inspects exported
graphs. Export dependencies
are pinned in [`scripts/requirements-lightonocr-export.txt`](../scripts/requirements-lightonocr-export.txt).
The inspector and exporter classify checkpoints from their configuration and
support the shared Qwen3/Pixtral full-attention architecture used by
LightOnOCR-2-1B and LightOnOCR-3-1B; model version is recorded as provenance,
not used to select an export implementation. The actual local export and
parity run documented here used the LightOnOCR-3-1B checkpoint.
ONNX files and `export_report.json` are generated under the ignored
`models/LightOnOCR-3-1B-ONNX/` directory, with graphs and external data in its
`onnx/` subdirectory. The exporter copies only `config.json`,
`processor_config.json`, `generation_config.json`, `tokenizer_config.json`,
`tokenizer.json`, and `chat_template.jinja`. The chat template is required for
processor prompt rendering; `special_tokens_map.json` is redundant with the
tokenizer assets. PyTorch safetensors and unrelated metadata are not copied.

The export used Python 3.11.7, Transformers 5.5.4, PyTorch 2.6.0, ONNX 1.17.0,
ONNX Runtime 1.20.1 CPU, opset 20, and float32 ONNX weights. Checkpoint weights
are BF16 and were loaded as float32 for this export. The graphs use IR version
9. The Transformers model was loaded with its `LightOnOcrForConditionalGeneration`
class while retaining the checkpoint's top-level `mistral3` config; Transformers
emits a wrapper/config-family warning for this supported class pairing.

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `vision_encoder.onnx` | `pixel_values` float32 `[batch, 3, height, width]` | `image_features` float32 `[batch, image_features, 1024]` |
| `embed_tokens.onnx` | `input_ids` int64 `[batch, sequence]` | `inputs_embeds` float32 `[batch, sequence, 1024]` |
| `decoder_model_merged.onnx` | `inputs_embeds` float32 `[batch, sequence, 1024]`, `attention_mask` int64 `[batch, total_sequence]`, and 56 float32 K/V tensors named `past_key_values.{0..27}.{key,value}` shaped `[batch, 8, past_sequence, 128]` | `logits` float32 `[batch, sequence, 151936]` and 56 float32 K/V tensors named `present.{0..27}.{key,value}` shaped `[batch, 8, total_sequence, 128]` |

The exported decoder derives text positions from the past sequence length and
has no separate position-ID input. The vision graph accepts the processor's
NCHW pixel tensor and derives the Pixtral patch grid from its dynamic height
and width; `image_sizes` is processor metadata, not a graph input. It fuses
the Pixtral encoder with the LightOnOCR projection so the output is already in
the decoder hidden space. This interface is close to the LightOnOCR-2 split,
with the 3-1B tokenizer/config adapter still required.

The 252×644 fixed fixture produces pixel values shaped `[1,3,252,644]` and
input IDs shaped `[1,219]`, including 207 image placeholders. The image tensor
SHA-256 is `bf03fdb3a986c099c8d43da9890edd2c9068d6b1f063aa2c04501c304597de9b`.
The vision graph also passed parity at 280×700 (250 image features), checking
that patch positions and merging generalize across height and width. ONNX
Runtime results matched Transformers for vision features, embeddings, prefill,
cached decode, and eight greedy generated tokens. The generated IDs were
`[687, 77915, 220, 16, 15, 19, 17, 2303]`, decoding to `INVOICE 1042` plus
trailing whitespace. Tolerances and full input token IDs are recorded in the
local `export_report.json`.

The decoder's external weights are in
`onnx/decoder_model_merged.onnx.data`; vision and embedding weights are stored
in their graph files. SHA-256 hashes for all four files are included in the
generated export report. CPU was the only
execution provider validated for this export.

## Repository baseline and design constraints

The existing engine loads three ONNX sessions (vision encoder, token embedding,
decoder), reads architecture fields from nested `text_config`, uses fixed
`LayerType::FullAttention`, and allocates conventional key/value cache tensors.
Its processor parses Pixtral image tokens and renders its own simplified
`<|im_start|>` chat template. The Python wrapper selects files using the
existing `default`, `fp16`, and `q4` preset name map. These are LightOnOCR-2
assumptions and cannot be applied unconditionally to `qwen3_5`.

Required design:

1. Parse a minimal top-level checkpoint descriptor first, including
   `model_type` and `architectures`, without deserializing architecture-only
   fields into the current `DecoderConfig`.
2. Dispatch by recognized architecture family (`lighton_ocr`/the verified
   LightOnOCR-2 family and `qwen3_5`) and reject unknown families with an error
   that includes the observed value and config path.
3. Keep the current LightOnOCR backend intact for old checkpoints. Place
   architecture-specific config parsing, processor, ONNX session wiring, and
   decoding state behind internal backend boundaries. Reuse tokenizer and
   runtime helpers only where asset and tensor contracts match.
4. Use graph input/output names, dtypes, ranks, and dynamic dimensions as the
   source of truth for exported-model execution. Config-derived dimensions are
   validation inputs, not substitutes for graph inspection.
5. Keep public `LightOnOCR::from_pretrained`, `LightOnOCROptions`, result types,
   Python `LightOnOCR.from_pretrained`, and default plain OCR behavior
   compatible. Architecture selection is automatic. Optional grounding prompt
   support should fit the existing prompt parameter if its semantics match;
   do not add a breaking required argument.
6. Keep checkpoint weights and large ONNX data files outside git. Store local
   test/export artifacts in an ignored model-assets directory or another
   explicitly documented artifact location, with revision/hash provenance.

## ONNX/export status and discovery checklist

The official LightOnOCR-3 model repositories currently expose Transformers
checkpoints and processor/tokenizer files; no official ONNX assets were used
for this export. Stage 1 can therefore be completed using the existing
LightOnOCR-2 ONNX repository. Qwen3.5 end-to-end tests wait until local 0.8B
and 4B exports exist. For the initial checkpoint inspection, the three
`model.safetensors` files (about 1.71 GB, 2.01 GB, and 9.08 GB) plus the
config/processor/tokenizer/template assets were downloaded from the Hugging
Face `main` revisions on 2026-11-20 to
`/private/tmp/lightonocr3-torch-inspection/`. Safetensors were inspected by
metadata (key, shape and dtype) without materializing all weights or running
inference. Their commit SHAs are pinned above for reproducible parity/export
runs; `main` is not a reproducible test reference.

Before implementing the Qwen3.5 backend, export small, inspectable graphs and
record for each model:

- exact graph filenames, external-data files, opset, ONNX IR version, and
  exporter/ORT versions;
- graph input/output names, ordering, element types, ranks, and symbolic/static
  dimensions for vision, embedding, prefill, and decode;
- whether the export is one combined model or separate vision, embedding, and
  decoder graphs;
- image preprocessing outputs, including pixel tensor layout/dtype, any
  `image_grid_thw`/grid metadata, image crops or patch packing, and resulting
  image token/feature counts;
- positional inputs (including position IDs and multimodal RoPE sections),
  cache input/output names, cache layout, and cache update semantics;
- whether linear-attention layers expose recurrent convolution/state tensors
  in addition to full-attention K/V tensors, and how state grows or is reused;
- logits shape/vocabulary, EOS IDs, stop conditions, and parity of one-token
  prefill and decode against Transformers;
- supported ORT execution providers and required custom operators, if any.

The repository's default Python environment has Transformers 4.48.3, which is
too old to load these checkpoints. The 3-1B export was therefore run in an
isolated Python 3.11.7 environment pinned by
`scripts/requirements-lightonocr-export.txt`; do not upgrade runtime
dependencies to satisfy this development-only reference path. Use a separately
pinned Transformers 5.x environment for Qwen3.5 numerical parity/export too.

Export a tiny smoke-test model or use reduced test inputs first to validate
operator support and session loading before spending time on full-size exports.
Then export each target architecture and precision with fixed opset and
reproducible command/configuration. Record SHA-256 hashes for model and
external-data files. Do not claim 0.8B/4B support until the actual exported
graphs load and complete a parity inference.

