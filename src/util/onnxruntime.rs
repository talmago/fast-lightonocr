//! ONNX Runtime compatibility validation.

#[cfg(feature = "load-dynamic")]
use std::ffi::c_void;
use std::ffi::{CStr, c_char};
#[cfg(feature = "load-dynamic")]
use std::path::PathBuf;

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
    Cuda {
        /// CUDA device index.
        device_id: usize,
    },
}

/// Ensures the configured ONNX Runtime is compatible with this crate.
pub(crate) fn ensure_compatible() -> Result<()> {
    runtime_details().and_then(|details| validate_runtime(&details))
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
