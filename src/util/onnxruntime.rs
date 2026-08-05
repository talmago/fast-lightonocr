//! ONNX Runtime compatibility validation and session configuration.

#[cfg(feature = "load-dynamic")]
use std::ffi::c_void;
use std::ffi::{CStr, c_char};
#[cfg(feature = "load-dynamic")]
use std::path::PathBuf;

use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

use crate::{Error, Result};

const REQUIRED_ORT_MAJOR: u32 = 1;
const REQUIRED_ORT_MINOR: u32 = 28;
const REQUIRED_ORT_API_LEVEL: u32 = ort::sys::ORT_API_VERSION;

/// ONNX Runtime execution provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU execution provider.
    #[default]
    Cpu,

    /// CUDA execution provider.
    ///
    /// Reserved for a future milestone. Selecting this provider currently
    /// returns an error at session creation time.
    Cuda {
        /// CUDA device index.
        device_id: usize,
    },
}

/// Runtime configuration applied when creating ONNX Runtime sessions.
///
/// The three LightOnOCR sessions (vision, embedding, decoder) run
/// sequentially for a single OCR request, so each session may use the full
/// `intra_threads` budget without oversubscribing during one request.
///
/// Note: Microsoft's prebuilt ONNX Runtime binaries are often built with
/// OpenMP. In that case `intra_threads` has no effect and thread count is
/// controlled by `OMP_NUM_THREADS` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// ONNX Runtime execution provider.
    pub execution_provider: ExecutionProvider,

    /// Intra-op thread count used to parallelize work within graph nodes.
    ///
    /// When `None`, defaults to the host's available parallelism.
    pub intra_threads: Option<usize>,

    /// Inter-op thread count used when parallel execution mode is enabled.
    ///
    /// When `None`, defaults to `1`. Has no effect while
    /// [`Self::parallel_execution`] is `false`.
    pub inter_threads: Option<usize>,

    /// Enable ORT parallel execution mode across independent graph nodes.
    ///
    /// Defaults to `false` (sequential execution mode).
    pub parallel_execution: bool,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
        }
    }
}

impl RuntimeOptions {
    /// Returns a copy with the given execution provider.
    #[must_use]
    pub fn with_execution_provider(mut self, execution_provider: ExecutionProvider) -> Self {
        self.execution_provider = execution_provider;
        self
    }

    /// Returns a copy with an explicit intra-op thread count.
    #[must_use]
    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }

    /// Returns a copy with an explicit inter-op thread count.
    #[must_use]
    pub fn with_inter_threads(mut self, inter_threads: usize) -> Self {
        self.inter_threads = Some(inter_threads);
        self
    }

    /// Returns a copy with parallel execution mode enabled or disabled.
    #[must_use]
    pub fn with_parallel_execution(mut self, parallel_execution: bool) -> Self {
        self.parallel_execution = parallel_execution;
        self
    }

    fn resolved_intra_threads(self) -> usize {
        self.intra_threads.unwrap_or_else(default_intra_threads)
    }

    fn resolved_inter_threads(self) -> usize {
        self.inter_threads.unwrap_or(1)
    }
}

fn default_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Ensures the configured ONNX Runtime is compatible with this crate.
pub(crate) fn ensure_compatible() -> Result<()> {
    runtime_details().and_then(|details| validate_runtime(&details))
}

/// Builds a session builder configured from [`RuntimeOptions`].
///
/// Callers should finish with [`SessionBuilder::commit_from_file`] and map
/// load failures to model-specific errors.
pub(crate) fn session_builder(options: &RuntimeOptions) -> Result<SessionBuilder> {
    ensure_compatible()?;
    validate_execution_provider(options.execution_provider)?;

    let intra_threads = options.resolved_intra_threads();
    let inter_threads = options.resolved_inter_threads();

    let builder = Session::builder().map_err(session_config_error)?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(session_config_error)?;
    let builder = builder
        .with_intra_threads(intra_threads)
        .map_err(session_config_error)?;
    let builder = builder
        .with_inter_threads(inter_threads)
        .map_err(session_config_error)?;
    builder
        .with_parallel_execution(options.parallel_execution)
        .map_err(session_config_error)
}

fn session_config_error<T>(source: ort::Error<T>) -> Error {
    Error::OnnxRuntimeCompatibility {
        reason: format!("failed to configure ONNX Runtime session: {source}"),
    }
}

fn validate_execution_provider(execution_provider: ExecutionProvider) -> Result<()> {
    match execution_provider {
        ExecutionProvider::Cpu => Ok(()),
        ExecutionProvider::Cuda { device_id } => Err(Error::OnnxRuntimeCompatibility {
            reason: format!(
                "CUDA execution provider (device_id={device_id}) is not wired yet; \
                 use ExecutionProvider::Cpu"
            ),
        }),
    }
}

#[derive(Debug)]
struct RuntimeDetails {
    version: String,
    has_required_api: bool,
    location: String,
}

#[cfg(not(feature = "load-dynamic"))]
fn runtime_details() -> Result<RuntimeDetails> {
    let base = unsafe { ort::sys::OrtGetApiBase() };
    if base.is_null() {
        return Err(Error::OnnxRuntimeCompatibility {
            reason: "OrtGetApiBase returned a null pointer".to_owned(),
        });
    }

    let version = version_string(unsafe { ((*base).GetVersionString)() })?;
    let api = unsafe { ((*base).GetApi)(REQUIRED_ORT_API_LEVEL) };

    Ok(RuntimeDetails {
        version,
        has_required_api: !api.is_null(),
        location: "linked ONNX Runtime library".to_owned(),
    })
}

#[cfg(feature = "load-dynamic")]
fn runtime_details() -> Result<RuntimeDetails> {
    let path = dynamic_library_path();
    let library = unsafe { libloading::Library::new(&path) }.map_err(|source| {
        Error::OnnxRuntimeCompatibility {
            reason: format!(
                "failed to load ONNX Runtime dynamic library {}: {source}",
                path.display()
            ),
        }
    })?;

    let get_api_base = unsafe {
        library.get::<unsafe extern "system" fn() -> *const OrtApiBaseCompat>(b"OrtGetApiBase")
    }
    .map_err(|source| Error::OnnxRuntimeCompatibility {
        reason: format!("{} does not export OrtGetApiBase: {source}", path.display()),
    })?;

    let base = unsafe { get_api_base() };
    if base.is_null() {
        return Err(Error::OnnxRuntimeCompatibility {
            reason: format!("{} returned a null OrtApiBase pointer", path.display()),
        });
    }

    let version = version_string(unsafe { ((*base).get_version_string)() })?;
    let api = unsafe { ((*base).get_api)(REQUIRED_ORT_API_LEVEL) };

    ort::init_from(&path).map_err(|source| Error::OnnxRuntimeCompatibility {
        reason: format!(
            "failed to initialize ONNX Runtime dynamic library {}: {source}",
            path.display()
        ),
    })?;

    Ok(RuntimeDetails {
        version,
        has_required_api: !api.is_null(),
        location: path.display().to_string(),
    })
}

#[cfg(feature = "load-dynamic")]
#[repr(C)]
struct OrtApiBaseCompat {
    get_api: unsafe extern "system" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "system" fn() -> *const c_char,
}

#[cfg(feature = "load-dynamic")]
fn dynamic_library_path() -> PathBuf {
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        #[cfg(target_os = "windows")]
        _ => PathBuf::from("onnxruntime.dll"),
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        _ => PathBuf::from("libonnxruntime.so"),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        _ => PathBuf::from("libonnxruntime.dylib"),
    }
}

fn version_string(version: *const c_char) -> Result<String> {
    if version.is_null() {
        return Err(Error::OnnxRuntimeCompatibility {
            reason: "GetVersionString returned a null pointer".to_owned(),
        });
    }

    Ok(unsafe { CStr::from_ptr(version) }
        .to_string_lossy()
        .into_owned())
}

fn validate_runtime(details: &RuntimeDetails) -> Result<()> {
    let (major, minor, _patch) = parse_version(&details.version)?;

    if major != REQUIRED_ORT_MAJOR || minor < REQUIRED_ORT_MINOR {
        return Err(Error::OnnxRuntimeCompatibility {
            reason: format!(
                "{} reports ONNX Runtime {}, but fast-lightonocr expects \
                 ONNX Runtime 1.28.x or a later 1.x release exposing C API \
                 level {REQUIRED_ORT_API_LEVEL}",
                details.location, details.version
            ),
        });
    }

    if !details.has_required_api {
        return Err(Error::OnnxRuntimeCompatibility {
            reason: format!(
                "{} reports ONNX Runtime {}, but does not expose required \
                 C API level {REQUIRED_ORT_API_LEVEL}",
                details.location, details.version
            ),
        });
    }

    Ok(())
}

fn parse_version(version: &str) -> Result<(u32, u32, u32)> {
    let mut parts = version.split('.');
    let major = parse_version_part(version, parts.next(), "major")?;
    let minor = parse_version_part(version, parts.next(), "minor")?;
    let patch = parts
        .next()
        .map(|value| {
            let numeric: String = value
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect();
            numeric.parse::<u32>()
        })
        .transpose()
        .map_err(|_| Error::OnnxRuntimeCompatibility {
            reason: format!("could not parse ONNX Runtime version {version:?}"),
        })?
        .unwrap_or(0);

    Ok((major, minor, patch))
}

fn parse_version_part(version: &str, part: Option<&str>, name: &str) -> Result<u32> {
    part.and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| Error::OnnxRuntimeCompatibility {
            reason: format!("could not parse ONNX Runtime {name} version from {version:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unwired_cuda_execution_provider() {
        let options = RuntimeOptions::default()
            .with_execution_provider(ExecutionProvider::Cuda { device_id: 0 });

        let error = validate_execution_provider(options.execution_provider)
            .expect_err("CUDA EP should be rejected until wired");

        match error {
            Error::OnnxRuntimeCompatibility { reason } => {
                assert!(reason.contains("CUDA"));
                assert!(reason.contains("not wired"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn resolves_default_thread_counts() {
        let options = RuntimeOptions::default();
        assert!(options.resolved_intra_threads() >= 1);
        assert_eq!(options.resolved_inter_threads(), 1);
        assert!(!options.parallel_execution);
    }
}
