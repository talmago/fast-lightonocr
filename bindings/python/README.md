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

```bash
pip install fast-lightonocr
```

Or with Poetry:

```bash
poetry add fast-lightonocr
```

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

Install the project with Poetry:

```bash
poetry install --with dev
```

Install the extension in editable mode:

```bash
maturin develop --release
```

Build a wheel:

```bash
maturin build --release
```

When using dynamic ONNX Runtime loading, set:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime
```

---

## 🙏 Acknowledgements

This package wraps the native Rust **Fast LightOnOCR** inference engine and
uses the open-weight **LightOnOCR** model released by Baidu.

- 🤗 https://huggingface.co/onnx-community/LightOnOCR-2-1B-ONNX
- 💻 https://github.com/baidu/LightOnOCR