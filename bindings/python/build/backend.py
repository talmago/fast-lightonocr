"""PEP 517 build backend for Fast LightOnOCR.

The backend delegates wheel construction to maturin, but first translates a
Python build profile into Cargo features and ONNX Runtime linker settings.
"""

from __future__ import annotations

import atexit
import ctypes
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import maturin

PROFILE_ENV = "BUILD_PROFILE"
PROFILE_CONFIG_KEYS = (
    "fast-lightonocr.profile",
    "fast_lightonocr.profile",
)

REQUIRED_ORT_MAJOR = 1
REQUIRED_ORT_MINOR = 28
REQUIRED_ORT_API_LEVEL = 27
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORT_DYLIB_PATH = "ORT_DYLIB_PATH"
ORT_LIB_PATH = "ORT_LIB_PATH"
ORT_LIB_LOCATION = "ORT_LIB_LOCATION"
ORT_PREFER_DYNAMIC_LINK = "ORT_PREFER_DYNAMIC_LINK"

SHARED_LIBRARY_PATH_ENV = {
    "Darwin": "DYLD_LIBRARY_PATH",
    "Linux": "LD_LIBRARY_PATH",
    "FreeBSD": "LD_LIBRARY_PATH",
    "Windows": "PATH",
}


@dataclass(frozen=True)
class BuildProfile:
    """Python build profile mapped to Cargo and runtime requirements."""

    name: str
    cargo_features: tuple[str, ...]
    runtime_distribution: str
    build_requirements: tuple[str, ...]


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a wheel through maturin with the selected profile configured."""

    settings = _profiled_config_settings(config_settings)
    with _configured_build_environment(settings):
        return maturin.build_wheel(wheel_directory, settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build an editable wheel through maturin."""

    settings = _profiled_config_settings(config_settings)
    with _configured_build_environment(settings):
        return maturin.build_editable(wheel_directory, settings, metadata_directory)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Prepare metadata using the same Cargo feature profile as wheel builds."""

    return maturin.prepare_metadata_for_build_wheel(
        metadata_directory,
        _profiled_config_settings(config_settings),
    )


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Build a source distribution through maturin."""

    return maturin.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return maturin and profile-specific build requirements."""

    requirements = maturin.get_requires_for_build_wheel(config_settings)
    settings = _profiled_config_settings(config_settings)

    if not _uses_load_dynamic(settings) and not _has_runtime_override():
        requirements.extend(_selected_profile(settings).build_requirements)

    return requirements


get_requires_for_build_editable = get_requires_for_build_wheel


def get_requires_for_build_sdist(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return maturin requirements for source distributions."""

    return maturin.get_requires_for_build_sdist(config_settings)


class _configured_build_environment:
    def __init__(self, config_settings: Mapping[str, Any] | None) -> None:
        self._settings = config_settings
        self._old_env: dict[str, str | None] = {}

    def __enter__(self) -> None:
        if _uses_load_dynamic(self._settings):
            return

        profile = _selected_profile(self._settings)
        runtime = _configured_onnx_runtime(profile)
        link_dir = _prepare_link_directory(runtime)
        self._set_env(ORT_LIB_PATH, str(link_dir))
        self._set_env(ORT_LIB_LOCATION, str(link_dir))
        self._set_env(ORT_PREFER_DYNAMIC_LINK, "1")
        self._prepend_library_path(link_dir)
        self._prepend_library_path(runtime.parent)

        print(
            "Using ONNX Runtime "
            f"{_runtime_version(runtime)} from {runtime} "
            f"for the fast-lightonocr {profile.name} build profile"
        )

    def __exit__(self, *_exc_info: object) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_env(self, key: str, value: str) -> None:
        if key not in self._old_env:
            self._old_env[key] = os.environ.get(key)
        os.environ[key] = value

    def _prepend_library_path(self, path: Path) -> None:
        key = SHARED_LIBRARY_PATH_ENV.get(platform.system())
        if key is None:
            return

        current = os.environ.get(key)
        value = str(path)
        if current:
            value = os.pathsep.join([value, current])
        self._set_env(key, value)


def _profiled_config_settings(
    config_settings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    settings = dict(config_settings or {})
    profile = _selected_profile(settings)
    args = maturin.get_maturin_pep517_args(settings)
    args = _merge_cargo_features(args, profile.cargo_features)
    settings["maturin.build-args"] = args
    return settings


def _selected_profile(config_settings: Mapping[str, Any] | None) -> BuildProfile:
    raw = os.environ.get(PROFILE_ENV)
    if raw is None:
        for key in PROFILE_CONFIG_KEYS:
            value = _config_value(config_settings, key)
            if value is not None:
                raw = value
                break

    profiles = _build_profiles()
    name = (raw or _default_profile_name()).strip().lower().replace("_", "-")
    try:
        return profiles[name]
    except KeyError as exc:
        expected = ", ".join(sorted(profiles))
        raise ValueError(
            f"unknown fast-lightonocr build profile {name!r}; "
            f"expected one of: {expected}"
        ) from exc


@lru_cache
def _build_profiles() -> dict[str, BuildProfile]:
    profile_configs = _fast_lightonocr_tool_config().get("build-profiles", {})
    profiles: dict[str, BuildProfile] = {}
    for raw_name, profile_config in profile_configs.items():
        name = raw_name.strip().lower().replace("_", "-")
        profiles[name] = BuildProfile(
            name=name,
            cargo_features=tuple(profile_config.get("cargo-features", ())),
            runtime_distribution=profile_config["runtime-distribution"],
            build_requirements=tuple(profile_config.get("build-requirements", ())),
        )

    if not profiles:
        raise RuntimeError(
            "pyproject.toml does not define fast-lightonocr build profiles"
        )

    return profiles


def _default_profile_name() -> str:
    build_config = _fast_lightonocr_tool_config().get("build", {})
    return str(build_config.get("default-profile", "cpu"))


@lru_cache
def _fast_lightonocr_tool_config() -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    return pyproject.get("tool", {}).get("fast-lightonocr", {})


def _config_value(
    config_settings: Mapping[str, Any] | None,
    key: str,
) -> str | None:
    if not config_settings or key not in config_settings:
        return None

    value = config_settings[key]
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and value:
        return str(value[-1])
    return str(value)


def _merge_cargo_features(
    args: list[str],
    cargo_features: tuple[str, ...],
) -> list[str]:
    if not cargo_features:
        return args

    merged = list(args)
    wanted = list(cargo_features)

    for index, arg in enumerate(merged):
        if arg == "--all-features":
            return merged
        if arg == "--features":
            if index + 1 >= len(merged):
                raise ValueError("maturin --features was passed without a feature list")
            merged[index + 1] = _format_features(
                _parse_features(merged[index + 1]) + wanted
            )
            return merged
        if arg.startswith("--features="):
            current = arg.split("=", 1)[1]
            merged[index] = "--features=" + _format_features(
                _parse_features(current) + wanted
            )
            return merged

    merged.extend(["--features", _format_features(wanted)])
    return merged


def _uses_load_dynamic(config_settings: Mapping[str, Any] | None) -> bool:
    args = maturin.get_maturin_pep517_args(config_settings)
    if "--all-features" in args:
        return True

    for index, arg in enumerate(args):
        if arg == "--features" and index + 1 < len(args):
            if "load-dynamic" in _parse_features(args[index + 1]):
                return True
        elif arg.startswith("--features="):
            if "load-dynamic" in _parse_features(arg.split("=", 1)[1]):
                return True

    return False


def _parse_features(value: str) -> list[str]:
    return [feature for feature in re.split(r"[\s,]+", value) if feature]


def _format_features(features: list[str]) -> str:
    return ",".join(dict.fromkeys(features))


def _has_runtime_override() -> bool:
    return any(
        os.environ.get(name)
        for name in (ORT_DYLIB_PATH, ORT_LIB_PATH, ORT_LIB_LOCATION)
    )


def _configured_onnx_runtime(profile: BuildProfile) -> Path:
    if dylib := os.environ.get(ORT_DYLIB_PATH):
        runtime = Path(dylib).expanduser()
        if not runtime.is_file():
            raise RuntimeError(
                f"{ORT_DYLIB_PATH} points to {runtime}, but that file does not exist"
            )
        return _validate_onnx_runtime(runtime)

    if lib_dir := os.environ.get(ORT_LIB_PATH) or os.environ.get(ORT_LIB_LOCATION):
        runtime = _find_runtime_library(Path(lib_dir).expanduser())
        return _validate_onnx_runtime(runtime)

    runtime = _discover_python_onnx_runtime(profile)
    return _validate_onnx_runtime(runtime)


def _discover_python_onnx_runtime(profile: BuildProfile) -> Path:
    try:
        importlib.metadata.version(profile.runtime_distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        requirement = (
            profile.build_requirements[0]
            if profile.build_requirements
            else profile.runtime_distribution
        )
        raise RuntimeError(
            "could not find an installed ONNX Runtime distribution for the "
            f"fast-lightonocr {profile.name} build profile. Install "
            f"{requirement!r}, install with the matching Python extra, or set "
            f"{ORT_DYLIB_PATH} to a compatible ONNX Runtime library."
        ) from exc

    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"{profile.runtime_distribution!r} is installed, but the "
            "'onnxruntime' package could not be located"
        )

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    return _find_runtime_library(package_dir / "capi")


def _find_runtime_library(directory: Path) -> Path:
    if not directory.is_dir():
        raise RuntimeError(
            "expected an ONNX Runtime library directory, "
            f"but {directory} is not a directory"
        )

    candidates = sorted(
        (path for path in directory.iterdir() if _is_runtime_library(path)),
        key=_library_sort_key,
    )
    if not candidates:
        raise RuntimeError(
            f"could not find an ONNX Runtime shared library in {directory}"
        )
    return candidates[0]


def _is_runtime_library(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name
    system = platform.system()
    if system == "Darwin":
        return name == "libonnxruntime.dylib" or (
            name.startswith("libonnxruntime.") and name.endswith(".dylib")
        )
    if system == "Windows":
        return name.lower() == "onnxruntime.dll"
    return name == "libonnxruntime.so" or name.startswith("libonnxruntime.so.")


def _library_sort_key(path: Path) -> tuple[int, str]:
    aliases = {
        "Darwin": "libonnxruntime.dylib",
        "Linux": "libonnxruntime.so",
        "FreeBSD": "libonnxruntime.so",
        "Windows": "onnxruntime.dll",
    }
    alias = aliases.get(platform.system())
    return (0 if path.name == alias else 1, path.name)


def _validate_onnx_runtime(runtime: Path) -> Path:
    version, has_required_api = _inspect_onnx_runtime(runtime)
    major, minor, _patch = _parse_version(version)

    if major != REQUIRED_ORT_MAJOR or minor < REQUIRED_ORT_MINOR:
        raise RuntimeError(
            f"ONNX Runtime {version} at {runtime} is not supported. "
            "fast-lightonocr expects ONNX Runtime 1.28.x or a later "
            f"1.x release that exposes C API level {REQUIRED_ORT_API_LEVEL}."
        )

    if not has_required_api:
        raise RuntimeError(
            f"ONNX Runtime {version} at {runtime} does not expose required "
            f"C API level {REQUIRED_ORT_API_LEVEL}. Use ONNX Runtime 1.28.x "
            "or a later compatible 1.x runtime."
        )

    return runtime


def _runtime_version(runtime: Path) -> str:
    version, _has_required_api = _inspect_onnx_runtime(runtime)
    return version


def _inspect_onnx_runtime(runtime: Path) -> tuple[str, bool]:
    try:
        loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
        library = loader(str(runtime))
    except OSError as exc:
        raise RuntimeError(
            f"failed to load ONNX Runtime library at {runtime}: {exc}"
        ) from exc

    calling_convention = (
        ctypes.WINFUNCTYPE if platform.system() == "Windows" else ctypes.CFUNCTYPE
    )

    class OrtApiBase(ctypes.Structure):
        _fields_ = [
            ("GetApi", calling_convention(ctypes.c_void_p, ctypes.c_uint32)),
            ("GetVersionString", calling_convention(ctypes.c_char_p)),
        ]

    try:
        get_api_base = library.OrtGetApiBase
    except AttributeError as exc:
        raise RuntimeError(
            f"{runtime} does not export OrtGetApiBase and is not an "
            "ONNX Runtime C API library"
        ) from exc

    get_api_base.restype = ctypes.c_void_p
    base_ptr = get_api_base()
    if not base_ptr:
        raise RuntimeError(f"{runtime} returned a null OrtApiBase pointer")

    base = ctypes.cast(base_ptr, ctypes.POINTER(OrtApiBase)).contents
    version_bytes = base.GetVersionString()
    version = version_bytes.decode("utf-8", errors="replace")
    has_required_api = bool(base.GetApi(REQUIRED_ORT_API_LEVEL))
    return version, has_required_api


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        raise RuntimeError(f"could not parse ONNX Runtime version string {version!r}")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return major, minor, patch


def _prepare_link_directory(runtime: Path) -> Path:
    link_dir = Path(tempfile.mkdtemp(prefix="fast-lightonocr-ort-"))
    atexit.register(shutil.rmtree, link_dir, ignore_errors=True)

    for path in runtime.parent.iterdir():
        if _should_copy_runtime_artifact(path):
            _link_or_copy(path, link_dir / path.name)

    for alias in _link_alias_names():
        _link_or_copy(runtime, link_dir / alias)

    if platform.system() == "Windows":
        import_library = runtime.with_suffix(".lib")
        if import_library.is_file():
            _link_or_copy(import_library, link_dir / import_library.name)
        elif not (link_dir / "onnxruntime.lib").is_file():
            raise RuntimeError(
                "ONNX Runtime on Windows requires onnxruntime.lib for "
                f"linking, but no import library was found next to {runtime}"
            )

    return link_dir


def _should_copy_runtime_artifact(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name.lower()
    if platform.system() == "Windows":
        return name.startswith("onnxruntime") and (
            name.endswith(".dll") or name.endswith(".lib")
        )

    return name.startswith("libonnxruntime") and (
        ".so" in name or name.endswith(".dylib")
    )


def _link_alias_names() -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return ("libonnxruntime.dylib", "libonnxruntime.1.dylib")
    if system in {"Linux", "FreeBSD"}:
        return ("libonnxruntime.so", "libonnxruntime.so.1")
    if system == "Windows":
        return ("onnxruntime.dll",)
    return ()


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return

    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
