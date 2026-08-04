# fast-lightonocr

> ⚡ Native Python bindings for the Rust **Fast LightOnOCR** inference engine.

`fast-lightonocr` provides high-performance OCR for documents and images using
Baidu's **LightOnOCR** model. Model inference runs entirely in native Rust,
while the Python package adds automatic Hugging Face downloads and structured
document parsing.

---

## ✨ Features

- 🚀 Native Rust inference engine
- 🧠 ONNX Runtime backend
- 📄 OCR for documents and images
- 📝 Structured Markdown output
- 📊 Structured HTML table extraction
- 🎨 Configurable table rendering
- 🎛️ Multiple model presets (`default`, `fp16`, `q4`)

---

## 📦 Installation

Install the package with the desired ONNX Runtime backend.

### CPU

```bash
pip install "fast-lightonocr[cpu]"
```

### CUDA

```bash
pip install "fast-lightonocr[cuda]"
```

> **Note**
>
> CUDA packaging is available through a dedicated build profile, although CUDA
> execution is not yet fully supported.

Prebuilt wheels are currently published for Linux x86_64 and macOS arm64. These
wheels bundle the required ONNX Runtime shared library, so no additional runtime
installation or environment configuration is required.

macOS x86_64 (Intel) wheels are not published because ONNX Runtime 1.28 does
not provide a compatible Python wheel for that platform.

### Building from source

When installing from source, the build backend automatically discovers a
compatible ONNX Runtime for the selected build profile.

If `ORT_DYLIB_PATH` is set, it is used directly. Otherwise, the build backend
installs the appropriate ONNX Runtime build dependency into the isolated build
environment, validates compatibility with ONNX Runtime 1.28.x (C API level 27),
configures Cargo automatically, and bundles the required native runtime library
into the resulting wheel.

---

## 🚀 Quick Start

```python
from fast_lightonocr import LightOnOCR

model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX",
)

result = model.process("receipt.jpg")
```

The first call downloads the required model files from Hugging Face and caches
them locally.

---

## 📄 OCR Results

The raw model output is available through `result.text`.

```python
print(result.text)
```

The Python bindings also expose a parsed document representation that extracts
embedded HTML tables while preserving the original document structure.

```python
print(result.document)
```

Tables can be accessed directly:

```python
for table in result.tables:
    print(table.text_rows)
```

---

## 📋 Table Rendering

By default, tables are rendered using ASCII borders.

```python
result = model.process(
    "receipt.jpg",
    table_format="grid",
)
```

Markdown tables are also supported.

```python
result = model.process(
    "receipt.jpg",
    table_format="github",
)
```

Any table format supported by `tabulate` may be used.

---

## ⚙️ Model Presets

`from_pretrained()` supports three ONNX model presets.

```python
model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX",
    preset="q4",
)
```

Available presets:

- `default`
- `fp16`
- `q4`

The generation length can be overridden:

```python
model = LightOnOCR.from_pretrained(
    "...",
    max_new_tokens=1024,
)
```

---

## 🛠 Development

Install the project and development dependencies:

```bash
poetry install --with dev
```

### Editable development

For local development, install the extension in editable mode with dynamic ONNX
Runtime loading:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime
poetry run maturin develop --release --features load-dynamic
```

For example, when using the Python `onnxruntime` package on macOS:

```bash
export ORT_DYLIB_PATH="$(python -c \
'import onnxruntime, pathlib; print(next((pathlib.Path(onnxruntime.__file__).parent / "capi").glob("libonnxruntime*.dylib")))')"
```

### Building a wheel

To build a distributable wheel, use the project's Python build backend:

```bash
poetry run pip wheel . --wheel-dir dist
```

The default build profile targets CPU execution and does not enable any Cargo
features. During source builds, the build backend automatically discovers a
compatible ONNX Runtime from `ORT_DYLIB_PATH` or from the selected build
profile's Python runtime package, validates compatibility with ONNX Runtime
1.28.x (C API level 27), configures Cargo, and produces a wheel containing the
required native runtime libraries.

To build using the CUDA profile:

```bash
BUILD_PROFILE=cuda poetry run pip wheel . --wheel-dir dist
```

> **Note**
>
> Running `maturin develop` **without** `--features load-dynamic` is not
> supported. The custom build backend is responsible for configuring ONNX
> Runtime linking during production builds, whereas editable development uses
> the `load-dynamic` feature together with `ORT_DYLIB_PATH`.

---

## 🙏 Acknowledgements

This package wraps the native Rust **Fast LightOnOCR** inference engine and
uses the open-weight **LightOnOCR** model released by Baidu.

- 🤗 https://huggingface.co/onnx-community/LightOnOCR-2-1B-ONNX
- 💻 https://github.com/baidu/LightOnOCR
