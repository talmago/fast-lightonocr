use std::path::{Path, PathBuf};
use std::process::Command;

use fast_lightonocr::Error;
use fast_lightonocr::Result;
use fast_lightonocr::model::ImageFeatures;
use fast_lightonocr::model::ImageTensor;
use fast_lightonocr::model::vision_encoder::{VisionConfig, VisionEncoder};
use fast_lightonocr::model::{DataType, ModelType};
use serde::Deserialize;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn vision_encoder_from_file(path: impl AsRef<Path>) -> Result<VisionEncoder> {
    let directory = path.as_ref();
    VisionEncoder::from_model_path(
        directory.join("onnx/vision_encoder_q4.onnx"),
        VisionConfig::from_file(directory.join("config.json"))?,
    )
}

#[test]
fn loads_vision_config_from_model_directory() {
    let config =
        VisionConfig::from_file(fixture_path("lightonocr_config").join("config.json")).unwrap();

    assert_eq!(config.model_type, ModelType::Pixtral);
    assert_eq!(config.dtype, DataType::BFloat16);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_channels, 3);
    assert_eq!(config.patch_size, 14);
}

#[test]
fn reports_missing_vision_model() {
    let error = vision_encoder_from_file(fixture_path("lightonocr_config"))
        .expect_err("vision encoder loading should fail without ONNX assets");

    match error {
        Error::MissingVisionModel { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("vision_encoder_q4.onnx")
            );
        }
        other => panic!("expected missing vision model error, got {other:?}"),
    }
}

#[test]
fn image_features_expose_shape_and_values() {
    let features = ImageFeatures::new(vec![0.0, 1.0, 2.0, 3.0], 1, 2, 2).unwrap();

    assert_eq!(features.shape(), (1, 2, 2));
    assert_eq!(features.batch_size(), 1);
    assert_eq!(features.num_merged_patches(), 2);
    assert_eq!(features.hidden_size(), 2);
    assert_eq!(features.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(features.get(0, 1, 1), Some(3.0));
    assert_eq!(features.get(1, 0, 0), None);
}

#[test]
fn rejects_invalid_image_feature_shapes() {
    let error =
        ImageFeatures::new(vec![0.0; 3], 1, 2, 2).expect_err("shape validation should fail");

    match error {
        Error::InvalidVisionOutput { reason } => {
            assert!(reason.contains("does not match shape"));
        }
        other => panic!("expected invalid vision output error, got {other:?}"),
    }
}

#[test]
fn official_vision_encoder_runs_when_assets_are_available() {
    let Ok(model_directory) = std::env::var("LIGHTONOCR_VISION_DIR") else {
        return;
    };

    let mut encoder = vision_encoder_from_file(&model_directory).unwrap();
    let hidden_size = encoder.config().hidden_size;
    let pixel_values = ImageTensor::new(vec![0.0; 3 * 14 * 14], 1, 3, 14, 14).unwrap();

    let features = encoder.encode(&pixel_values).unwrap();

    assert_eq!(features.batch_size(), 1);
    assert_eq!(features.hidden_size(), hidden_size);
    assert!(!features.as_slice().is_empty());
}

#[test]
fn python_vision_encoder_parity_when_enabled() {
    if std::env::var("LIGHTONOCR_RUN_PYTHON_VISION_PARITY").is_err() {
        return;
    }

    let model_directory =
        std::env::var("LIGHTONOCR_VISION_DIR").expect("LIGHTONOCR_VISION_DIR is required");
    let mut encoder = vision_encoder_from_file(&model_directory).unwrap();
    let pixel_values = ImageTensor::new(vec![0.0; 3 * 14 * 14], 1, 3, 14, 14).unwrap();
    let rust_features = encoder.encode(&pixel_values).unwrap();

    let script = format!(
        r#"
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

model_dir = Path({model_directory:?})
model_path = model_dir / "onnx" / "vision_encoder_q4.onnx"
pixel_values = np.zeros((1, 3, 14, 14), dtype=np.float32)
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
output = session.run(["image_features"], {{"pixel_values": pixel_values}})[0]
print(json.dumps({{"shape": list(output.shape), "data": output.reshape(-1).tolist()}}))
"#,
        model_directory = model_directory
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run python3");
    assert!(
        output.status.success(),
        "python parity script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let expected: PythonVisionOutput =
        serde_json::from_slice(&output.stdout).expect("invalid python JSON output");
    assert_eq!(rust_features.shape(), expected.shape_tuple());
    assert_eq!(rust_features.as_slice().len(), expected.data.len());
    for (actual, expected) in rust_features.as_slice().iter().zip(expected.data.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "feature mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug, Deserialize)]
struct PythonVisionOutput {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl PythonVisionOutput {
    fn shape_tuple(&self) -> (usize, usize, usize) {
        assert_eq!(self.shape.len(), 3);
        (self.shape[0], self.shape[1], self.shape[2])
    }
}
