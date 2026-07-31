use std::path::{Path, PathBuf};
use std::process::Command;

use fast_lightonocr::Error;
use fast_lightonocr::Result;
use fast_lightonocr::model::InputEmbeddings;
use fast_lightonocr::model::config::{DataType, ModelType};
use fast_lightonocr::model::embedding::{EmbeddingConfig, EmbeddingModel};
use serde::Deserialize;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn embedding_model_from_file(path: impl AsRef<Path>) -> Result<EmbeddingModel> {
    let directory = path.as_ref();
    EmbeddingModel::from_model_path(
        directory.join("onnx/embed_tokens_q4.onnx"),
        EmbeddingConfig::from_file(directory.join("config.json"))?,
    )
}

#[test]
fn loads_embedding_config_from_model_directory() {
    let config =
        EmbeddingConfig::from_file(fixture_path("lightonocr_config").join("config.json")).unwrap();

    assert_eq!(config.model_type, ModelType::Qwen3);
    assert_eq!(config.dtype, DataType::BFloat16);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.vocab_size, 151936);
}

#[test]
fn reports_missing_embedding_model() {
    let error = embedding_model_from_file(fixture_path("lightonocr_config"))
        .expect_err("embedding model loading should fail without ONNX assets");

    match error {
        Error::MissingEmbeddingModel { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("embed_tokens_q4.onnx")
            );
        }
        other => panic!("expected missing embedding model error, got {other:?}"),
    }
}

#[test]
fn input_embeddings_expose_shape_and_values() {
    let embeddings = InputEmbeddings::new(vec![0.0, 1.0, 2.0, 3.0], 1, 2, 2).unwrap();

    assert_eq!(embeddings.shape(), (1, 2, 2));
    assert_eq!(embeddings.batch_size(), 1);
    assert_eq!(embeddings.sequence_length(), 2);
    assert_eq!(embeddings.hidden_size(), 2);
    assert_eq!(embeddings.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(embeddings.get(0, 1, 1), Some(3.0));
    assert_eq!(embeddings.get(1, 0, 0), None);
}

#[test]
fn rejects_invalid_input_embedding_shapes() {
    let error =
        InputEmbeddings::new(vec![0.0; 3], 1, 2, 2).expect_err("shape validation should fail");

    match error {
        Error::InvalidEmbeddingOutput { reason } => {
            assert!(reason.contains("does not match shape"));
        }
        other => panic!("expected invalid embedding output error, got {other:?}"),
    }
}

#[test]
fn official_embedding_model_runs_when_assets_are_available() {
    let Ok(model_directory) = std::env::var("LIGHTONOCR_EMBEDDING_DIR") else {
        return;
    };

    let mut model = embedding_model_from_file(&model_directory).unwrap();
    let hidden_size = model.config().hidden_size;
    let input_ids = vec![0, 1, 2];

    let embeddings = model.embed(&input_ids).unwrap();

    assert_eq!(embeddings.shape(), (1, input_ids.len(), hidden_size));
    assert!(!embeddings.as_slice().is_empty());
}

#[test]
fn python_embedding_model_parity_when_enabled() {
    if std::env::var("LIGHTONOCR_RUN_PYTHON_EMBEDDING_PARITY").is_err() {
        return;
    }

    let model_directory =
        std::env::var("LIGHTONOCR_EMBEDDING_DIR").expect("LIGHTONOCR_EMBEDDING_DIR is required");
    let mut model = embedding_model_from_file(&model_directory).unwrap();
    let input_ids = vec![0, 1, 2];
    let rust_embeddings = model.embed(&input_ids).unwrap();

    let script = format!(
        r#"
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

model_dir = Path({model_directory:?})
model_path = model_dir / "onnx" / "embed_tokens_q4.onnx"
input_ids = np.array([[0, 1, 2]], dtype=np.int64)
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
output = session.run(["inputs_embeds"], {{"input_ids": input_ids}})[0]
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

    let expected: PythonEmbeddingOutput =
        serde_json::from_slice(&output.stdout).expect("invalid python JSON output");
    assert_eq!(rust_embeddings.shape(), expected.shape_tuple());
    assert_eq!(rust_embeddings.as_slice().len(), expected.data.len());
    for (actual, expected) in rust_embeddings.as_slice().iter().zip(expected.data.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "embedding mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug, Deserialize)]
struct PythonEmbeddingOutput {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl PythonEmbeddingOutput {
    fn shape_tuple(&self) -> (usize, usize, usize) {
        assert_eq!(self.shape.len(), 3);
        (self.shape[0], self.shape[1], self.shape[2])
    }
}
