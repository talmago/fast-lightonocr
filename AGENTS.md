# AGENTS.md

> This is the primary entry point for AI coding agents. Read this file first before making changes to the repository.

---

# Repository Overview

**Fast LightOnOCR** is a native Rust inference engine for the **LightOnOCR-2-1B** vision-language model.

The project executes the official ONNX models directly through **ONNX Runtime**, providing a lightweight OCR engine without requiring Python, PyTorch, or Hugging Face Transformers at runtime.

The implementation aims to be functionally equivalent to the official Python implementation while providing an idiomatic Rust API, a command-line interface, and optional Python bindings.

---

# Documentation

Project documentation lives under:

```text
docs/
```

Before modifying a subsystem, read its corresponding documentation.

| Document | Purpose |
|----------|---------|
| `ARCHITECTURE.md` | High-level system architecture |
| `ROADMAP.md` | Development roadmap and milestones |
| `MODEL_CONTRACTS.md` | ONNX model interfaces and tensor contracts |
| `KV.md` | Decoder KV cache (host vs CUDA) and GPU decode plan |

Avoid duplicating documentation across multiple files. High-level concepts belong in `ARCHITECTURE.md`, while subsystem-specific implementation details belong in their dedicated documents.

---

# Development Principles

## Correctness First

Behavior should match the official Hugging Face implementation whenever practical.

When behavior is unclear, prefer matching the reference implementation over introducing new behavior.

Parity testing is considered a core part of development.

---

## Native Rust

Do not introduce runtime dependencies on:

- Python
- PyTorch
- Transformers
- NumPy

Python may be used only for development tools, model inspection, parity testing, and reference scripts.

---

## Preserve ONNX Contracts

Do not modify the exported ONNX models.

Adapt the Rust implementation to the published model interfaces rather than introducing model-specific assumptions or workarounds.

---

## Strong Typing

Prefer domain-specific Rust types over loosely typed structures.

Avoid passing raw tensors, JSON values, or untyped collections between components unless required by ONNX Runtime.

---

## Small, Focused Modules

Each module should have a single responsibility.

Prefer small reusable components over large monolithic implementations.

---

## Explicit Data Flow

Tensor transformations should remain explicit.

Avoid hidden reshaping, implicit conversions, or unnecessary abstraction layers.

---

## Public API Stability

Internal implementation details may evolve.

The public API should remain small, consistent, and easy to understand.

---

## Model Variants

Different model variants (for example FP16 and Q4) should share the same inference engine.

Differences between variants should be expressed through configuration rather than conditional logic throughout the codebase.

---

## Keep It Simple

This project targets a single family of models.

Prefer straightforward, model-specific implementations over generic abstractions unless they provide a clear long-term benefit.

---

# Testing

Whenever practical, compare Rust outputs against the official Python implementation.

Validation should include, where applicable:

- tokenizer outputs
- image preprocessing
- intermediate tensor values
- ONNX model outputs
- generated token IDs
- final OCR text

Prefer parity tests over implementation-specific tests.

---

# Coding Style

Follow standard Rust conventions.

Prefer:

- `Result<T, Error>` for fallible APIs
- `thiserror` for error definitions
- `serde` for configuration loading
- descriptive type names
- exhaustive documentation for public APIs

Run before committing:

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```

---

# Python Bindings

Python bindings are located under:

```text
python/
```

The bindings should remain a thin wrapper around the Rust library.

All inference logic belongs in the Rust implementation.

---

# Scope

This repository is dedicated exclusively to **LightOnOCR** inference.

Do not introduce generic inference frameworks or unnecessary abstraction layers unless they provide a clear benefit to the project.

Favor maintainability, readability, and correctness over premature optimization.

---

# Goal

The primary objective is to build a production-quality native Rust inference engine that is:

- functionally equivalent to the official implementation
- lightweight
- deterministic
- modular
- easy to embed
- extensible to future LightOnOCR model variants