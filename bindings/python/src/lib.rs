//! PyO3 bindings for Fast LightOnOCR.

use std::path::PathBuf;
use std::sync::Mutex;

use fast_lightonocr::{
    FinishReason, LightOnOCR as NativeLightOnOCR, LightOnOCROptions, OCRResult as NativeOCRResult,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

/// OCR inference result.
#[pyclass(name = "OCRResult", module = "fast_lightonocr._native")]
#[derive(Debug, Clone)]
struct PyOCRResult {
    text: String,
    token_ids: Vec<i64>,
    finish_reason: String,
}

#[pymethods]
impl PyOCRResult {
    /// Decoded OCR text.
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    /// Generated token IDs.
    #[getter]
    fn token_ids(&self) -> Vec<i64> {
        self.token_ids.clone()
    }

    /// Reason generation stopped.
    #[getter]
    fn finish_reason(&self) -> &str {
        &self.finish_reason
    }

    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "OCRResult(text={:?}, token_ids={}, finish_reason={:?})",
            self.text,
            self.token_ids.len(),
            self.finish_reason
        )
    }
}

impl From<NativeOCRResult> for PyOCRResult {
    fn from(result: NativeOCRResult) -> Self {
        let finish_reason = match result.finish_reason() {
            FinishReason::EndOfSequence => "end_of_sequence",
            FinishReason::Length => "length",
        }
        .to_owned();

        Self {
            text: result.text().to_owned(),
            token_ids: result.token_ids().to_vec(),
            finish_reason,
        }
    }
}

/// Native LightOnOCR inference engine.
#[pyclass(name = "LightOnOCR", module = "fast_lightonocr._native")]
struct PyLightOnOCR {
    inner: Mutex<NativeLightOnOCR>,
}

#[pymethods]
impl PyLightOnOCR {
    /// Loads a model from a local Hugging Face model directory.
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "_load_model_dir")]
    #[pyo3(signature = (
        model_dir,
        *,
        preset = "default",
        max_new_tokens = None,
        vision_encoder = None,
        embedding = None,
        decoder = None
    ))]
    fn load_model_dir(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model_dir: PathBuf,
        preset: &str,
        max_new_tokens: Option<usize>,
        vision_encoder: Option<PathBuf>,
        embedding: Option<PathBuf>,
        decoder: Option<PathBuf>,
    ) -> PyResult<Self> {
        let options =
            options_from_args(preset, max_new_tokens, vision_encoder, embedding, decoder)?;
        let model = py.allow_threads(|| {
            NativeLightOnOCR::from_pretrained(&model_dir, options).map_err(to_py_err)
        })?;

        Ok(Self {
            inner: Mutex::new(model),
        })
    }

    /// Loads and processes an image from disk.
    #[pyo3(signature = (image_path, system_prompt = None))]
    fn process(
        &self,
        py: Python<'_>,
        image_path: PathBuf,
        system_prompt: Option<String>,
    ) -> PyResult<PyOCRResult> {
        let result = py.allow_threads(|| {
            let mut model = self.inner.lock().map_err(|_| {
                PyRuntimeError::new_err("LightOnOCR instance is unavailable after a panic")
            })?;

            model
                .process_file(&image_path, system_prompt.as_deref())
                .map(PyOCRResult::from)
                .map_err(to_py_err)
        })?;

        Ok(result)
    }
}

fn options_from_args(
    preset: &str,
    max_new_tokens: Option<usize>,
    vision_encoder: Option<PathBuf>,
    embedding: Option<PathBuf>,
    decoder: Option<PathBuf>,
) -> PyResult<LightOnOCROptions> {
    let mut options = match preset {
        "default" => LightOnOCROptions::default(),
        "fp16" => LightOnOCROptions::fp16(),
        "q4" => LightOnOCROptions::q4(),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown model preset {other:?}; expected 'default', 'fp16', or 'q4'"
            )));
        }
    };

    if let Some(max_new_tokens) = max_new_tokens {
        options.max_new_tokens = Some(max_new_tokens);
    }
    if let Some(vision_encoder) = vision_encoder {
        options.vision_encoder = vision_encoder;
    }
    if let Some(embedding) = embedding {
        options.embedding = embedding;
    }
    if let Some(decoder) = decoder {
        options.decoder = decoder;
    }

    Ok(options)
}

fn to_py_err(error: fast_lightonocr::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Python extension module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLightOnOCR>()?;
    module.add_class::<PyOCRResult>()?;
    Ok(())
}
