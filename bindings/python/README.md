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

Install with the matching extra for your backend. Published wheels target
**Linux x86_64** and **macOS arm64** (macOS Intel is not published: ONNX
Runtime 1.28 has no compatible wheel there).

### CPU (default)

```bash
pip install "fast-lightonocr[cpu]"
```

CPU wheels bundle ONNX Runtime. No extra environment setup is required.

### CUDA

```bash
pip install "fast-lightonocr[cuda]"
```

Requires a CUDA-enabled package build and a compatible NVIDIA driver. The
`cuda` extra pulls in `onnxruntime-gpu` (CUDA 13 / cuDNN) and `nvidia-cublas`.

Select CUDA at load time:

```python
model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX",
    runtime_kwargs={
        "execution_provider": "cuda",
        "device_id": 0,
    },
)
```

When `execution_provider="cuda"`, `from_pretrained` preloads the pip NVIDIA
CUDA/cuDNN libraries (`onnxruntime.preload_dlls`), so `LD_LIBRARY_PATH` is
usually unnecessary. CPU loads never take that path.

### Building from source

Source installs use the project build backend. It discovers ONNX Runtime from
`ORT_DYLIB_PATH` when set, otherwise from the profile’s Python ORT package,
validates ONNX Runtime 1.28.x (C API level 27), and bundles the native runtime
into the wheel.

#### CPU

```bash
pip install -v ".[cpu]"
```

# or explicitly:

```bash
BUILD_PROFILE=cpu pip install -v ".[cpu]"
```

#### CUDA

```bash
BUILD_PROFILE=cuda pip install -v ".[cuda]"
```

`BUILD_PROFILE=cuda` enables the native `cuda` Cargo feature and injects the
ORT CUDA provider plugins (`libonnxruntime_providers_{shared,cuda}`) into the
wheel. The `[cuda]` extra installs the CUDA 13 / cuDNN / cublas user
libraries used at runtime.

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

### Generation overrides

Model defaults come from Hugging Face `generation_config.json` (typically
`do_sample=True`, `temperature=0.2`, `top_k=0`, `top_p=0.9`). 

Override them at load time with `generation_kwargs` (merged onto the decoder config; unknown keys raise `ValueError`):

```python
# Faster / deterministic OCR on CPU (greedy decoding)
model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX",
    preset="q4",
    generation_kwargs={
        "do_sample": False,
        "max_new_tokens": 256,
    },
)

# Sampling with a top-k cutoff (HF default top_k=0 walks the full vocab)
model = LightOnOCR.from_pretrained(
    "...",
    generation_kwargs={
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 0.9,
        "max_new_tokens": 256,
    },
)
```

Supported keys: `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`.

You can also update knobs after load:

```python
model.generation_kwargs = {"do_sample": False}
print(model.generation_kwargs)
```

Bare `max_new_tokens=` remains supported as a shorthand:

```python
model = LightOnOCR.from_pretrained("...", max_new_tokens=1024)
```

> On CPU, prefer `do_sample=False` for throughput.

> If you need sampling, set a modest `top_k` (for example `50`) instead of leaving the HF default `top_k=0`.

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

Same profiles as [Building from source](#building-from-source):

```bash
# CPU (default)
poetry run pip wheel . --wheel-dir dist

# CUDA
BUILD_PROFILE=cuda poetry run pip wheel . --wheel-dir dist
```

> **Note**
>
> Running `maturin develop` **without** `--features load-dynamic` is not
> supported. Production/`pip install` builds use the custom build backend for
> ONNX Runtime linking; editable development uses `load-dynamic` with
> `ORT_DYLIB_PATH`.

---

## 🙏 Acknowledgements

This package wraps the native Rust **Fast LightOnOCR** inference engine and
uses the open-weight **LightOnOCR** model released by Baidu.

- 🤗 https://huggingface.co/onnx-community/LightOnOCR-2-1B-ONNX
- 💻 https://github.com/baidu/LightOnOCR
