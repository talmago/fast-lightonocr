//! PyO3 bindings for Fast LightOnOCR.

use std::path::PathBuf;
use std::sync::Mutex;

use fast_lightonocr::{
    ExecutionProvider, FinishReason, GenerationConfig, LightOnOCR as NativeLightOnOCR,
    LightOnOCROptions, OCRResult as NativeOCRResult, RuntimeOptions,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

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
        generation_kwargs = None,
        runtime_kwargs = None,
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
        generation_kwargs: Option<&Bound<'_, PyDict>>,
        runtime_kwargs: Option<&Bound<'_, PyDict>>,
        max_new_tokens: Option<usize>,
        vision_encoder: Option<PathBuf>,
        embedding: Option<PathBuf>,
        decoder: Option<PathBuf>,
    ) -> PyResult<Self> {
        // Runtime options must be applied before load (sessions are created then).
        // Generation overrides are applied onto GenerationConfig after load so
        // LightOnOCROptions does not grow a parallel override type.
        let mut options = options_from_args(preset, vision_encoder, embedding, decoder)?;
        options.runtime = apply_runtime_kwargs(options.runtime, runtime_kwargs)?;
        let mut model = py.allow_threads(|| {
            NativeLightOnOCR::from_pretrained(&model_dir, options).map_err(to_py_err)
        })?;

        apply_generation_kwargs(
            model.generation_config_mut(),
            generation_kwargs,
            max_new_tokens,
        )?;

        Ok(Self {
            inner: Mutex::new(model),
        })
    }

    /// Current generation knobs as a dict (`max_new_tokens`, `do_sample`, …).
    #[getter]
    fn generation_kwargs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let model = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("LightOnOCR instance is unavailable after a panic")
        })?;
        generation_config_to_dict(py, model.generation_config())
    }

    /// Merge generation overrides onto the loaded decoder config.
    #[setter]
    fn set_generation_kwargs(&self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut model = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("LightOnOCR instance is unavailable after a panic")
        })?;
        apply_generation_kwargs(model.generation_config_mut(), Some(kwargs), None)
    }

    /// Runtime options used when the model sessions were created (read-only).
    #[getter]
    fn runtime_kwargs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let model = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("LightOnOCR instance is unavailable after a panic")
        })?;
        runtime_options_to_dict(py, model.runtime())
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

/// Applies runtime overrides onto [`RuntimeOptions`] before model load.
fn apply_runtime_kwargs(
    mut options: RuntimeOptions,
    runtime_kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<RuntimeOptions> {
    let Some(kwargs) = runtime_kwargs else {
        return Ok(options);
    };

    let mut device_id = match options.execution_provider {
        ExecutionProvider::Cuda { device_id } => device_id,
        ExecutionProvider::Cpu => 0,
    };
    let mut provider = match options.execution_provider {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::Cuda { .. } => "cuda",
    }
    .to_owned();

    for (key, value) in kwargs.iter() {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "execution_provider" => {
                provider = value.extract::<String>().map_err(|_| {
                    PyValueError::new_err(
                        "runtime_kwargs['execution_provider'] must be a string ('cpu' or 'cuda')",
                    )
                })?;
            }
            "device_id" => {
                device_id = extract_usize(&value, "runtime_kwargs", "device_id")?;
            }
            "intra_threads" => {
                options.intra_threads =
                    Some(extract_usize(&value, "runtime_kwargs", "intra_threads")?);
            }
            "inter_threads" => {
                options.inter_threads =
                    Some(extract_usize(&value, "runtime_kwargs", "inter_threads")?);
            }
            "parallel_execution" => {
                options.parallel_execution = value.extract::<bool>().map_err(|_| {
                    PyValueError::new_err("runtime_kwargs['parallel_execution'] must be a bool")
                })?;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported runtime_kwargs key {other:?}; expected one of: \
                     execution_provider, device_id, intra_threads, inter_threads, \
                     parallel_execution"
                )));
            }
        }
    }

    options.execution_provider = match provider.as_str() {
        "cpu" => ExecutionProvider::Cpu,
        "cuda" => ExecutionProvider::Cuda { device_id },
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported execution_provider {other:?}; expected 'cpu' or 'cuda'"
            )));
        }
    };

    Ok(options)
}

fn runtime_options_to_dict<'py>(
    py: Python<'py>,
    options: RuntimeOptions,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match options.execution_provider {
        ExecutionProvider::Cpu => {
            dict.set_item("execution_provider", "cpu")?;
            dict.set_item("device_id", 0)?;
        }
        ExecutionProvider::Cuda { device_id } => {
            dict.set_item("execution_provider", "cuda")?;
            dict.set_item("device_id", device_id)?;
        }
    }
    match options.intra_threads {
        Some(value) => dict.set_item("intra_threads", value)?,
        None => dict.set_item("intra_threads", py.None())?,
    }
    match options.inter_threads {
        Some(value) => dict.set_item("inter_threads", value)?,
        None => dict.set_item("inter_threads", py.None())?,
    }
    dict.set_item("parallel_execution", options.parallel_execution)?;
    Ok(dict)
}

/// Applies generation overrides onto an existing [`GenerationConfig`].
///
/// Keys present in `generation_kwargs` win over the bare `max_new_tokens`
/// alias when both are provided.
fn apply_generation_kwargs(
    config: &mut GenerationConfig,
    generation_kwargs: Option<&Bound<'_, PyDict>>,
    max_new_tokens_alias: Option<usize>,
) -> PyResult<()> {
    let mut applied_max_new_tokens = false;

    if let Some(kwargs) = generation_kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "max_new_tokens" => {
                    config.max_new_tokens =
                        extract_usize(&value, "generation_kwargs", "max_new_tokens")?;
                    applied_max_new_tokens = true;
                }
                "do_sample" => {
                    config.do_sample = value.extract::<bool>().map_err(|_| {
                        PyValueError::new_err("generation_kwargs['do_sample'] must be a bool")
                    })?;
                }
                "temperature" => {
                    config.temperature = value.extract::<f32>().map_err(|_| {
                        PyValueError::new_err("generation_kwargs['temperature'] must be a float")
                    })?;
                }
                "top_k" => {
                    config.top_k = extract_u32(&value, "generation_kwargs", "top_k")?;
                }
                "top_p" => {
                    config.top_p = value.extract::<f32>().map_err(|_| {
                        PyValueError::new_err("generation_kwargs['top_p'] must be a float")
                    })?;
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported generation_kwargs key {other:?}; expected one of: \
                         max_new_tokens, do_sample, temperature, top_k, top_p"
                    )));
                }
            }
        }
    }

    if !applied_max_new_tokens && let Some(max_new_tokens) = max_new_tokens_alias {
        config.max_new_tokens = max_new_tokens;
    }

    Ok(())
}

fn generation_config_to_dict<'py>(
    py: Python<'py>,
    config: &GenerationConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("max_new_tokens", config.max_new_tokens)?;
    dict.set_item("do_sample", config.do_sample)?;
    dict.set_item("temperature", config.temperature)?;
    dict.set_item("top_k", config.top_k)?;
    dict.set_item("top_p", config.top_p)?;
    Ok(dict)
}

fn extract_usize(value: &Bound<'_, PyAny>, kwargs_name: &str, key: &str) -> PyResult<usize> {
    value.extract::<usize>().map_err(|_| {
        PyValueError::new_err(format!("{kwargs_name}['{key}'] must be a non-negative int"))
    })
}

fn extract_u32(value: &Bound<'_, PyAny>, kwargs_name: &str, key: &str) -> PyResult<u32> {
    value.extract::<u32>().map_err(|_| {
        PyValueError::new_err(format!("{kwargs_name}['{key}'] must be a non-negative int"))
    })
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
