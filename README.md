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

[![Fast LightOnOCR Demo](docs/images/demo-thumbnail.png)](https://github.com/user-attachments/assets/5c7a1048-34df-4ac5-8a9a-2de5f646a090)

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

The recommended development workflow links against an existing ONNX Runtime installation.

Configure the runtime location:

```bash
export ORT_LIB_PATH=/path/to/onnxruntime
export ORT_PREFER_DYNAMIC_LINK=1

# macOS only
export DYLD_LIBRARY_PATH="$ORT_LIB_PATH"
```

Then build normally:

```bash
cargo build
```

Alternatively:

- `--features download-binaries` downloads a compatible ONNX Runtime automatically.
- `--features load-dynamic` loads ONNX Runtime explicitly at runtime.

### Test

```bash
cargo test
```

To use explicit runtime loading:

```bash
cargo test --features load-dynamic
```

### Lint

```bash
cargo clippy --all-targets --all-features
```

### Format

```bash
cargo fmt
```

### Run the examples

The inference example defaults to:

- model directory: `models/lightonocr`
- image: `examples/SROIE-receipt.jpeg`
- model preset: `default`

```bash
cargo run --example inference
```

Optional arguments are accepted in this order:

```bash
cargo run --example inference -- \
  <model-dir> <image-path> <default|fp16|q4>
```

To print decoded output as tokens are generated:

```bash
cargo run --example streaming
```

The streaming example accepts the same first three optional arguments, plus an optional generation limit:

```bash
cargo run --example streaming -- \
  <model-dir> <image-path> <default|fp16|q4> <max-new-tokens>
```

If using the runtime-loading feature:

```bash
cargo run --features load-dynamic --example inference
cargo run --features load-dynamic --example streaming
```

### Python Bindings

Python bindings are implemented using PyO3 and maturin.

For installation, packaging, development workflow, build profiles, and ONNX Runtime configuration, see:

- **[bindings/python/README.md](bindings/python/README.md)**

### ONNX Runtime

The project supports three runtime configurations:

| Mode | Description |
|------|-------------|
| **System runtime (recommended)** | Links against an existing ONNX Runtime installation using `ORT_LIB_PATH` and `ORT_PREFER_DYNAMIC_LINK`. |
| **download-binaries** | Downloads a compatible ONNX Runtime automatically during the build. |
| **load-dynamic** | Loads ONNX Runtime explicitly at runtime using `ORT_DYLIB_PATH`. |

The Python bindings use the same native library and build infrastructure.

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
