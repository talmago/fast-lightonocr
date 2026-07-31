# MODEL_CONTRACTS.md

# Overview

This document defines the runtime contracts of the LightOnOCR ONNX models.

The Rust inference engine treats each exported ONNX model as a black box with a well-defined interface. The responsibility of the Rust implementation is to provide correctly shaped input tensors, execute the models, and interpret their outputs.

The ONNX models themselves are never modified.

This document serves as the authoritative reference for all model interfaces used by the runtime.

---

# Goals

The model contracts provide:

* A specification of every ONNX model interface.
* Input and output tensor definitions.
* Tensor shapes and data types.
* Runtime assumptions.
* Configuration values derived from the exported models.
* A reference for parity testing against the official Python implementation.

---

# Runtime Overview

LightOnOCR inference consists of three ONNX models executed sequentially.

```text
                    Image
                      │
                      ▼
              Vision Encoder
                      │
             image_features
                      │
                      │
Prompt ──► Tokenizer ─┤
                      │
                      ▼
             Embedding Model
                      │
             inputs_embeds
                      │
                      ▼
            Embedding Merger
                      │
                      ▼
                 Decoder
                      │
        logits + KV Cache
```

The runtime is responsible for orchestrating these models while preserving the contracts defined below.

---

# Vision Encoder

## Purpose

Converts a preprocessed image into visual feature embeddings consumed by the language model.

## Inputs

| Name           | Type      | Shape                            |
| -------------- | --------- | -------------------------------- |
| `pixel_values` | `float32` | `(batch_size, 3, height, width)` |

## Outputs

| Name             | Type      | Shape                                    |
| ---------------- | --------- | ---------------------------------------- |
| `image_features` | `float32` | `(batch_size, num_merged_patches, 1024)` |

## Notes

The output sequence length depends on the preprocessing pipeline and is represented by the dynamic dimension `num_merged_patches`.

The hidden dimension is fixed at **1024**.

---

# Embedding Model

## Purpose

Converts token IDs into language model embeddings.

## Inputs

| Name        | Type    | Shape                           |
| ----------- | ------- | ------------------------------- |
| `input_ids` | `int64` | `(batch_size, sequence_length)` |

## Outputs

| Name            | Type      | Shape                                 |
| --------------- | --------- | ------------------------------------- |
| `inputs_embeds` | `float32` | `(batch_size, sequence_length, 1024)` |

## Notes

The embedding dimension matches the hidden size of the decoder (**1024**).

---

# Embedding Merge

The embedding merger is implemented in Rust.

It combines:

* image feature embeddings from the Vision Encoder
* token embeddings from the Embedding Model

by replacing image placeholder token embeddings with the corresponding image feature vectors.

The merged embedding sequence becomes the decoder input.

This step is not implemented by an ONNX model.

---

# Decoder

## Purpose

Performs autoregressive language generation.

The decoder consumes token embeddings together with an attention mask and the current KV cache, producing the next-token logits and an updated KV cache.

## Inputs

### Input Embeddings

| Name            | Type      | Shape                                 |
| --------------- | --------- | ------------------------------------- |
| `inputs_embeds` | `float32` | `(batch_size, sequence_length, 1024)` |

### Attention Mask

| Name             | Type    | Shape                                 |
| ---------------- | ------- | ------------------------------------- |
| `attention_mask` | `int64` | `(batch_size, total_sequence_length)` |

### KV Cache

Each decoder layer receives:

| Name                            | Type      | Shape                                        |
| ------------------------------- | --------- | -------------------------------------------- |
| `past_key_values.<layer>.key`   | `float32` | `(batch_size, 8, past_sequence_length, 128)` |
| `past_key_values.<layer>.value` | `float32` | `(batch_size, 8, past_sequence_length, 128)` |

The exported model contains **28 decoder layers**, each with one key tensor and one value tensor.

## Outputs

### Logits

| Name     | Type      | Shape                                   |
| -------- | --------- | --------------------------------------- |
| `logits` | `float32` | `(batch_size, sequence_length, 151936)` |

### Updated KV Cache

For every decoder layer the model returns:

| Name                    | Type      | Shape                                         |
| ----------------------- | --------- | --------------------------------------------- |
| `present.<layer>.key`   | `float32` | `(batch_size, 8, total_sequence_length, 128)` |
| `present.<layer>.value` | `float32` | `(batch_size, 8, total_sequence_length, 128)` |

---

# Runtime Constants

The exported ONNX models establish the following runtime characteristics.

| Property        |  Value |
| --------------- | -----: |
| Hidden size     |   1024 |
| Decoder layers  |     28 |
| KV heads        |      8 |
| Head dimension  |    128 |
| Vocabulary size | 151936 |

These values should be derived from the model configuration whenever practical, rather than duplicated in source code.

---

# Dynamic Dimensions

The exported models make use of symbolic dimensions.

| Dimension               | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `batch_size`            | Number of input samples                                    |
| `height`                | Preprocessed image height                                  |
| `width`                 | Preprocessed image width                                   |
| `sequence_length`       | Current decoder input length                               |
| `past_sequence_length`  | Existing KV cache length                                   |
| `total_sequence_length` | Total sequence length after decoding                       |
| `num_merged_patches`    | Number of visual embeddings produced by the Vision Encoder |

The runtime should avoid assuming fixed values for these dimensions unless explicitly required by the model.

---

# Model Metadata

| Property   | Value        |
| ---------- | ------------ |
| IR Version | 10           |
| ONNX Opset | 21           |
| Producer   | Hugging Face |

All exported models share the same metadata.

---

# Runtime Responsibilities

The Rust runtime is responsible for:

* loading the exported ONNX models
* validating model inputs
* executing model inference
* constructing decoder inputs
* managing the KV cache
* merging image and token embeddings
* exposing strongly typed model interfaces

The runtime should hide ONNX-specific details from the remainder of the codebase.

---

# Validation

Each ONNX model should be validated independently before integrating the complete OCR pipeline.

Validation should include:

* successful model loading
* tensor shape validation
* tensor type validation
* parity against the official Python implementation
* intermediate tensor comparison
* end-to-end OCR output comparison

Whenever practical, parity testing should compare intermediate tensors rather than only the final generated text.

---

# External Assets

The runtime depends on the following model assets:

* Vision Encoder ONNX model
* Embedding Model ONNX model
* Decoder ONNX model
* tokenizer configuration
* processor configuration
* model configuration
* generation configuration

These assets should be treated as immutable runtime dependencies.

---

# Design Principles

The runtime should follow these principles:

* Treat every ONNX model as a black box.
* Preserve the published model contracts.
* Use strongly typed Rust abstractions instead of raw tensors where possible.
* Derive runtime behavior from configuration files whenever practical.
* Validate inputs before model execution.
* Maintain parity with the official Python implementation.

