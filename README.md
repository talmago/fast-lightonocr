# fast-lightonocr

> ⚡ Native Rust inference engine for Baidu's **LightOnOCR** model using ONNX Runtime.

---

## ✨ Features

- 🚀 Native Rust inference
- 🧠 ONNX Runtime backend
- 📄 End-to-end OCR for documents and images
- 📝 Structured Markdown output
- 🎛️ Multiple model presets (FP16, Q4)
- 📥 Built-in model download utility
- 💻 CPU-first execution
- 🔌 Optional dynamic loading of ONNX Runtime
- 🐍 Python bindings with Hugging Face model download support

---

## 📦 Installation

### Rust

> **Note**
>
> The crate is currently under active development and has not yet been published on crates.io.

Clone the repository:

```bash
git clone https://github.com/talmago/fast-lightonocr.git
cd fast-lightonocr
```

Build the library:

```bash
cargo build
```

Download the official LightOnOCR model:

```bash
python scripts/download_model.py
```

### Python

The project also provides Python bindings with automatic model download and
structured document parsing.

Install from PyPI:

```bash
pip install fast-lightonocr
```

For installation options, build profiles, and the complete Python API, see:

- **[bindings/python/README.md](bindings/python/README.md)**

---

## 🚀 Usage

### Rust

```rust
use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

fn main() -> fast_lightonocr::Result<()> {
    let mut model = LightOnOCR::from_pretrained(
        "models/lightonocr",
        LightOnOCROptions::default(),
    )?;

    let result = model.process_file("receipt.jpg", None)?;

    println!("{}", result.text());

    Ok(())
}
```

Available model presets:

- `LightOnOCROptions::default()`
- `LightOnOCROptions::fp16()`
- `LightOnOCROptions::q4()`

`LightOnOCROptions::max_new_tokens` can be used to override the value loaded
from `generation_config.json`.

### Python

The Python bindings wrap the same native Rust engine and can automatically
download model assets from Hugging Face.

```python
from fast_lightonocr import LightOnOCR

model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX"
)

result = model.process("receipt.jpg")

# Raw OCR output (Markdown with embedded HTML tables)
print(result.text)

# Parsed document with rendered tables
print(result.document)

# Structured table extraction
for table in result.tables:
    print(table.text_rows)
```

The document parser automatically extracts embedded HTML tables while preserving
the original document order. Tables are rendered using `tabulate` and can be
configured through the `table_format` argument:

```python
result = model.process(
    "receipt.jpg",
    table_format="github",   # Markdown tables
)

result = model.process(
    "receipt.jpg",
    table_format="grid",     # ASCII tables (default)
)
```

---

## 🛠 Development

### Build

```bash
cargo build
```

### Run the example

```bash
cargo run --features load-dynamic --example inference
```

The example defaults to:

- model directory: `models/lightonocr`
- image: `examples/SROIE-receipt.jpeg`
- model preset: `default`

Optional arguments are accepted in this order:

```bash
cargo run --features load-dynamic --example inference -- \
  <model-dir> <image-path> <default|fp16|q4>
```

To print decoded output as tokens are generated, run the streaming example:

```bash
cargo run --features load-dynamic --example streaming
```

The streaming example accepts the same first three optional arguments as the
inference example, plus an optional generation limit:

```bash
cargo run --features load-dynamic --example streaming -- \
  <model-dir> <image-path> <default|fp16|q4> <max-new-tokens>
```

### Test

```bash
cargo test
```

### Lint

```bash
cargo clippy --all-targets --all-features
```

### Format

```bash
cargo fmt
```

### Python Bindings

Python bindings built with PyO3 and maturin.

For installation, development, packaging, build profiles, and ONNX Runtime configuration, see:

- **[bindings/python/README.md](bindings/python/README.md)**

### ONNX Runtime Discovery

The project supports two approaches for locating ONNX Runtime:

1. **Default builds** use the ONNX Runtime configured by the selected build profile.
2. **Dynamic loading** (enabled with the Rust `load-dynamic` feature) loads the runtime specified by the `ORT_DYLIB_PATH` environment variable.

When using `load-dynamic`, configure the runtime library location before building or running the project:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
```

The build tooling validates that the discovered runtime is compatible with the
version expected by the project (currently ONNX Runtime 1.28.x / C API level 27).

---

## 🗺 Roadmap

- ✅ Native Rust inference
- ✅ ONNX model execution
- ✅ Image preprocessing
- ✅ Autoregressive generation
- ✅ Sampling (temperature, top-p, top-k)
- ✅ Streaming generation example
- ✅ FP16 and Q4 model presets
- ✅ Python bindings
- 🚧 Generation parity and deterministic seeded generation
- 🚧 Broader processor parity coverage
- 🚧 Performance optimizations
- 🚧 Native CLI
- 🚧 Python packaging hardening

---

## 📚 Documentation

Additional project documentation is available in:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — architecture overview, model pipeline, and design decisions.
- **[MODEL_CONTRACTS.md](docs/MODEL_CONTRACTS.md)** — ONNX model interfaces and tensor contracts.
- **[ROADMAP.md](docs/ROADMAP.md)** — implementation milestones and future work.
- **[AGENTS.md](AGENTS.md)** — development guidelines for AI coding agents and contributors.

---

## 🙏 Acknowledgements

This project builds upon the open-weight **LightOnOCR** model released by Baidu.

- 🤗 [Hugging Face](https://huggingface.co/onnx-community/LightOnOCR-2-1B-ONNX)
- 💻 [Original project](https://github.com/baidu/LightOnOCR)

This repository provides a native Rust inference engine and does not include the original model weights.
