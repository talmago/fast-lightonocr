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
- ⚡ ~31 tok/s E2E on Apple Silicon CPU (Q4, greedy, 256 tokens; see `inference_bench`)
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

- **[python/README.md](python/README.md)**

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
from `generation_config.json`. CPU session tuning (intra/inter-op threads,
parallel execution) is configured via `RuntimeOptions` on
`LightOnOCROptions::runtime`.

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
  <model-dir> <image-path> <default|fp16|q4> <cpu|cuda> <device-id>
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

ORT session options (`execution_provider`, `intra_threads`, `inter_threads`,
parallel execution) are configured through `RuntimeOptions` on
`LightOnOCROptions`. Defaults use the CPU provider and host parallelism for
intra-op threads.

CUDA requires `--features cuda` (and a CUDA-enabled ONNX Runtime; ort 2.0.0-rc.13
targets CUDA 13 / cuDNN 9.x). Example:

```bash
cargo run --features load-dynamic,cuda --example inference -- \
  models/lightonocr examples/SROIE-receipt.jpeg default cuda
```

Optional 5th argument is the CUDA `device_id` (default `0`). On CUDA, decode
defaults to device-resident KV via IoBinding (`CudaKVCache`); set
`FAST_LIGHTONOCR_CUDA_HOST_KV` to force the host `KVCache` path. Sampling still
runs on the host. See [`docs/KV.md`](docs/KV.md).

To compare CPU thread settings on your machine:

```bash
cargo run --release --features load-dynamic --example cpu_ort_bench -- \
  models/lightonocr q4
```

For end-to-end latency and tokens/sec across presets (`q4` / `default` / `fp16`), generation lengths, and greedy vs sample:

```bash
cargo run --release --features load-dynamic --example inference_bench -- \
  models/lightonocr examples/SROIE-receipt.jpeg
```

Optional third argument selects presets (comma-separated), e.g. `q4` or `q4,default`.

`tok_s` is E2E-normalized (`tokens / process_file` seconds), so vision/prefill
cost is included. Numbers vary by machine, ORT build, thread settings, image,
and decoding mode; the Features callout (~31 tok/s) is a representative Apple
Silicon CPU result for Q4 greedy at 256 new tokens.

When ONNX Runtime is built with OpenMP, prefer `OMP_NUM_THREADS` over `intra_threads`.

### Python Bindings

Python bindings are implemented using PyO3 and maturin.

For installation, packaging, development workflow, build profiles, and ONNX Runtime configuration, see:

- **[python/README.md](python/README.md)**

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
- ✅ Python bindings and packaging (wheels, release workflow)
- ✅ Native CLI-style examples (`inference`, `streaming`)
- ✅ CPU performance work (KV-cache reuse, top-k/top-p, ORT session tuning, decode host reuse, inference bench)
- 🚧 Generation parity and deterministic seeded generation
- 🚧 Broader processor parity coverage
- ✅ CUDA execution provider + `KVCacheBackend` (`KVCache` / `CudaKVCache`)
- 🚧 CoreML / DirectML execution providers
- 🚧 Python exposure of runtime / EP options

See **[ROADMAP.md](docs/ROADMAP.md)** for milestone detail.

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
