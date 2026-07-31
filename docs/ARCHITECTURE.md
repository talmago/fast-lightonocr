# ARCHITECTURE.md

# Fast LightOnOCR

**Fast LightOnOCR** is a native Rust inference engine for the **LightOnOCR-2-1B** vision-language model.

The project executes the official ONNX models directly through **ONNX Runtime**, providing a lightweight, production-ready OCR engine without requiring Python, PyTorch, or the Hugging Face Transformers runtime.

The long-term goal is to provide a reusable Rust library, a command-line interface, and optional Python bindings, all built on the same inference engine.

---

# Goals

The project aims to:

- Execute the official LightOnOCR ONNX models natively in Rust.
- Eliminate the runtime dependency on Python and PyTorch.
- Produce results equivalent to the official Hugging Face implementation.
- Provide a clean, idiomatic Rust API.
- Support CPU inference as the primary deployment target.
- Expose optional Python bindings without introducing Python inference dependencies.
- Keep the architecture modular and easy to extend.

---

# Non-Goals

The project intentionally does **not** aim to:

- Reimplement or train the underlying neural network.
- Replace ONNX Runtime with a custom execution engine.
- Become a generic multimodal inference framework.
- Modify or optimize the original model weights.

The project focuses exclusively on efficient inference for the LightOnOCR family of models.

---

# High-Level Pipeline

The inference pipeline consists of three ONNX models together with the Rust
processor, embedding merge step, decoder generation loop, and tokenizer-backed
final decoding.

```text
Input image + optional prompt
        │
        ▼
Processor
- image preprocessing
- chat template rendering
- tokenization
- image-placeholder expansion
        │
        ├── pixel_values ───────────────► VisionEncoder
        │                                   │
        │                                   ▼
        │                              ImageFeatures
        │
        ├── input_ids ─────────────────► EmbeddingModel
        │                                   │
        │                                   ▼
        │                              InputEmbeddings
        │
        └── attention_mask

InputEmbeddings + ImageFeatures + image_token_id
        │
        ▼
merge_image_features()
        │
        ▼
Decoder::generate()
- decoder ONNX execution
- autoregressive loop
- KV cache
- stopping criteria
        │
        ▼
generated token IDs
        │
        ▼
decode_result()
        │
        ▼
OCR text
```

---

# System Components

The inference engine is organized into a small number of focused components.

## Configuration

Loads and validates the model, tokenizer, processor, and generation configuration files.

These configuration objects define model dimensions, special tokens, processor settings, and generation defaults used throughout the inference pipeline.

---

## Processor

The processor converts user-provided images and optional prompt text into
model-ready inputs.

It owns:

- an image processor;
- a text processor;
- the tokenizer;
- `processor_config.json` metadata.

Image preprocessing is driven by the exported `processor_config.json` and is
implemented to remain functionally equivalent to the official Hugging Face
Pixtral processor.

Typical operations include:

- image loading
- RGB conversion
- resizing while preserving the expected aspect ratio
- alignment to `patch_size × spatial_merge_size`
- pixel normalization
- padding to the target batch resolution
- conversion to an `ImageTensor`

Text processing applies the LightOnOCR chat-template layout, tokenizes the
rendered prompt, expands image placeholders according to the processed image
grid, and creates the `AttentionMask`.

---

## Tokenizer

The tokenizer converts rendered prompts into token IDs and decodes generated
token IDs back into text.

Its responsibilities include:

- encoding prompts into token IDs
- decoding generated token IDs

The tokenizer loads the exported `tokenizer.json` and is implemented to remain functionally equivalent to the official Hugging Face tokenizer.

Although LightOnOCR is a multimodal model, the tokenizer is responsible only
for textual tokenization and decoding. During inference, image placeholder
embeddings are replaced by the output of the vision encoder before the decoder
is executed.

---

## Prompt Expansion

LightOnOCR follows the Hugging Face Pixtral/LightOnOCR processor model:
image-placeholder expansion happens before embedding lookup.

The text processor first renders a logical prompt containing one
`<|image_pad|>` placeholder. For the default image-only OCR request, this
matches the Python reference:

```text
<|im_start|>system<|im_end|>
<|im_start|>user
<|image_pad|><|im_end|>
<|im_start|>assistant
```

After tokenization, the processor replaces that single image placeholder token
with a concrete vision-token grid derived from the processed image dimensions:

```text
patches_x = width / patch_size
patches_y = height / patch_size

grid_x = patches_x / spatial_merge_size
grid_y = patches_y / spatial_merge_size
```

The processed image dimensions are aligned to `patch_size × spatial_merge_size`
so that both patch counts divide evenly before prompt expansion.

The expanded image token sequence is:

```text
grid_x × <|image_pad|>
<|vision_pad|>
grid_x × <|image_pad|>
<|vision_pad|>
...
grid_x × <|image_pad|>
<|vision_end|>
```

For the reference SROIE example, the Python processor produces:

```text
pixel_values: (1, 3, 476, 476)
input_ids:    318 tokens
image_pad:    289 tokens
vision_pad:   16 tokens
image_features: (1, 289, 1024)
```

The `<|image_pad|>` count matches the number of vision features. Row separator
tokens and the final `<|vision_end|>` token remain ordinary text embeddings.

This design keeps sequence length fixed after tokenization. The embedding
merge step only replaces embeddings at `<|image_pad|>` positions; it does not
insert or remove tokens.

---

## ONNX Models

Fast LightOnOCR executes three ONNX models:

- Vision Encoder
- Token Embedding Model
- Decoder

Together they implement the complete multimodal inference pipeline.

Detailed model interfaces, tensor shapes, dynamic dimensions, and ONNX contracts are documented in **[MODEL_CONTRACTS.md](MODEL_CONTRACTS.md)**.

---

## Embedding Merger

`merge_image_features()` replaces image placeholder embeddings with the vision
encoder output, producing the multimodal embedding sequence consumed by the
decoder.

It validates:

- batch size;
- token sequence length versus embedding sequence length;
- image-placeholder count versus vision feature count;
- hidden-size compatibility.

The image placeholder token ID comes from `Decoder::image_token_id()`, which is
loaded from `config.json`.

---

## Decoder

The decoder owns the ONNX decoder session, decoder configuration, generation
configuration, autoregressive generation loop, and KV cache.

Responsibilities include:

- validating decoder inputs
- executing the decoder ONNX graph
- KV-cache management
- attention mask updates
- token selection
- stopping criteria

`Decoder::generate()` receives the merged initial `InputEmbeddings`, the
initial `AttentionMask`, and the embedding model used to embed each generated
next token. It returns generated token IDs and a finish reason indicating
whether generation stopped at EOS or at the configured length limit.

---

## ONNX Runtime

Model execution is delegated entirely to **ONNX Runtime**.

The inference engine owns one ONNX Runtime session for each exported model:

- Vision Encoder
- Embedding Model
- Decoder

Sessions are created during model initialization and reused for the lifetime of the inference engine.

The runtime provides lightweight, strongly typed wrappers around the exported ONNX models, exposing Rust domain types such as `ImageTensor`, `ImageFeatures`, `InputEmbeddings`, `AttentionMask`, `Logits`, and `KvCache` rather than raw ONNX tensors.

Runtime behavior—including model selection, execution providers, and session configuration—is encapsulated behind the inference engine, allowing higher-level components to remain independent of ONNX Runtime implementation details.

---

# Public API

The public API intentionally remains small.

A typical application only needs to load a model and perform OCR.

```rust
use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

let mut model = LightOnOCR::from_pretrained(
    "models/lightonocr",
    LightOnOCROptions::default(),
)?;

let result = model.process_file("receipt.jpg", None)?;

println!("{}", result.text());
```

Advanced configuration remains optional.

---

# Project Structure

```text
fast-lightonocr/
├── docs/
├── examples/
├── scripts/
├── src/
│   ├── model/
│   ├── processor/
│   ├── tokenizer/
│   ├── util/
│   └── lib.rs
├── tests/
└── Cargo.toml
```

The implementation is organized by responsibility rather than deployment target.

---

# Design Principles

## Correctness First

The Rust implementation should produce outputs equivalent to the official Hugging Face implementation.

Parity testing is considered a core part of development.

---

## Native Rust

The inference engine should not require Python, PyTorch, NumPy, or Transformers at runtime.

---

## Thin Wrappers

The three ONNX models remain unchanged.

Rust components provide lightweight wrappers around the official model interfaces rather than introducing unnecessary abstraction.

---

## Modular Components

Each stage of inference should be independently testable.

This enables isolated parity testing, simplifies debugging, and keeps individual components reusable.

---

## Explicit Data Flow

Tensor transformations remain explicit throughout the pipeline.

The implementation avoids hidden conversions and implicit tensor reshaping wherever possible.

---

## Extensibility

Although initially focused on LightOnOCR, the architecture should make it straightforward to support future model variants through configuration rather than architectural changes.

---

# Documentation

Additional implementation details are documented separately:

- **[MODEL_CONTRACTS.md](MODEL_CONTRACTS.md)** — ONNX model interfaces, tensor shapes, and contracts.
- **[ROADMAP.md](ROADMAP.md)** — implementation milestones and future work.
- **[AGENTS.md](../AGENTS.md)** — development guidelines for AI coding agents.
