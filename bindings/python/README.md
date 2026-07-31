# Fast LightOnOCR Python Bindings

Python bindings for the native Rust `fast-lightonocr` inference engine.

The Python package is intentionally thin:

- model inference runs in Rust;
- packaging is handled by PyO3 and maturin;
- Hugging Face model downloads are handled in Python through `huggingface_hub`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin
```

If you are using dynamic ONNX Runtime loading, set `ORT_DYLIB_PATH` before
loading a model:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
```

## Editable Development Install

From the repository root:

```bash
cd bindings/python
maturin develop --release
```

Then verify the package imports:

```bash
python -c "import fast_lightonocr"
```

## Build A Wheel

From the repository root:

```bash
cd bindings/python
maturin build --release
```

The wheel is written under the workspace `target/wheels/` directory.

## Run Rust Validation

From the repository root:

```bash
cargo check --features load-dynamic
cargo test --features load-dynamic
```

For the full workspace, including the Python binding crate:

```bash
cargo clippy --workspace --all-targets --all-features
```

## Python Usage

`LightOnOCR.from_pretrained()` is the single public model-loading API. It
accepts either a Hugging Face repository ID or a path to an existing local
model directory.

For a Hugging Face model, it downloads only the configuration, tokenizer,
processor, and ONNX files required by the selected preset:

```python
from fast_lightonocr import LightOnOCR

model = LightOnOCR.from_pretrained(
    "onnx-community/LightOnOCR-2-1B-ONNX",
)

result = model.process("document.png")

print(result.text)
print(result.clean_text)
print(result.tables)
print(result.token_ids)
print(result.finish_reason)
```

The same API loads an existing local model directory without downloading:

```python
from fast_lightonocr import LightOnOCR

model = LightOnOCR.from_pretrained("models/lightonocr")
result = model.process("examples/SROIE-receipt.jpeg")

print(result.clean_text)
```

Optional arguments:

```python
model = LightOnOCR.from_pretrained(
    model_id_or_path,
    revision=None,
    cache_dir=None,
    local_files_only=False,
    preset="default",
    max_new_tokens=None,
)
```

Supported presets are:

- `default`
- `fp16`
- `q4`

## OCR Results

`result.text` is always the raw text returned by the native Rust engine.

The Python wrapper also exposes pure Python post-processing:

- `result.clean_text`: the full OCR text with HTML tags stripped.
- `result.tables`: HTML tables parsed into lightweight Python objects.

Post-processing runs eagerly by default. To delay it until `clean_text` or
`tables` is accessed:

```python
result = model.process("document.png", post_process=False)
```

Each table exposes its original HTML and plain text rows:

```python
for table in result.tables:
    print(table.html)
    print(table.text_rows)
```

For example, processing `examples/SROIE-receipt.jpeg` produces table rows
similar to:

```text
['CODE/DESC', 'PRICE', 'Disc', 'AMOUNT']
['QTY', 'RM', '', 'RM']
['9556939040118', 'KF MODELLING CLAY KIDDY FISH', '', '']
['1 PC *', '9.000', '0.00', '9.00']
['Total :', '9.00']
['Rounding Adjustment :', '0.00']
['Rounded Total (RM):', '9.00']
```

The complete example is:

```python
from fast_lightonocr import LightOnOCR

model = LightOnOCR.from_pretrained("models/lightonocr")
result = model.process("examples/SROIE-receipt.jpeg")

print(result.text)       # Raw OCR output, including HTML tables.
print(result.clean_text) # Raw text with HTML tags removed.

for table in result.tables:
    for row in table.text_rows:
        print(row)
```
