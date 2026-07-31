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
- 🐍 Planned Python bindings

---

## 📦 Installation

> **Note**
>
> The crate is currently under active development and has not yet been published on crates.io.

Clone the repository and download the model:

```bash
git clone https://github.com/<org>/fast-lightonocr.git
cd fast-lightonocr

python scripts/download_model.py
```

This downloads the official ONNX model into:

```text
models/lightonocr/
```

---

## 🚀 Usage

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

`LightOnOCROptions::max_new_tokens` can be set to override the value loaded
from `generation_config.json`.

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

### Dynamic ONNX Runtime

On macOS, the recommended approach is to dynamically load ONNX Runtime.

Set the runtime library location:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
```

For example:

```bash
export ORT_DYLIB_PATH=$HOME/.pyenv/versions/3.13.9/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.28.0.dylib
```

---

## 🗺 Roadmap

- ✅ Native Rust inference
- ✅ ONNX model execution
- ✅ Image preprocessing
- ✅ Autoregressive generation
- ✅ Sampling (temperature, top-p, top-k)
- ✅ Streaming generation example
- ✅ FP16 and Q4 model presets
- 🚧 Generation parity and deterministic seeded generation
- 🚧 Broader processor parity coverage
- 🚧 Performance optimizations
- 🚧 Native CLI
- 🚧 Python bindings

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
