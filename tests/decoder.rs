use std::path::{Path, PathBuf};
use std::process::Command;

use fast_lightonocr::Error;
use fast_lightonocr::Result;
use fast_lightonocr::model::Logits;
use fast_lightonocr::model::decoder::{Decoder, DecoderConfig, KvCache, LayerCache, LayerType};
use fast_lightonocr::model::{AttentionMask, InputEmbeddings};
use fast_lightonocr::model::{DataType, ModelType};
use serde::Deserialize;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn decoder_from_file(path: impl AsRef<Path>) -> Result<Decoder> {
    let directory = path.as_ref();
    Decoder::from_model_path(directory.join("onnx/decoder_model_merged_q4.onnx"))
}

#[test]
fn loads_decoder_config_from_model_directory() {
    let config =
        DecoderConfig::from_file(fixture_path("lightonocr_config").join("config.json")).unwrap();

    assert_eq!(config.model_type, ModelType::Qwen3);
    assert_eq!(config.dtype, DataType::BFloat16);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 151936);
    assert_eq!(
        config.layer_types,
        vec![LayerType::FullAttention, LayerType::FullAttention]
    );
}

#[test]
fn reports_missing_decoder_model() {
    let error = decoder_from_file(fixture_path("lightonocr_config"))
        .expect_err("decoder loading should fail without ONNX assets");

    match error {
        Error::MissingDecoderModel { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("decoder_model_merged_q4.onnx")
            );
        }
        other => panic!("expected missing decoder model error, got {other:?}"),
    }
}

#[test]
fn initializes_empty_kv_cache_from_decoder_config() {
    let config =
        DecoderConfig::from_file(fixture_path("lightonocr_config").join("config.json")).unwrap();
    let cache = KvCache::empty(&config, 1).unwrap();

    assert!(cache.is_empty());
    assert_eq!(cache.layer_count(), 28);
    assert_eq!(cache.shape(), (1, 8, 0, 128));
    assert!(cache.layers().iter().all(|layer| layer.key().is_empty()));
    assert!(cache.layers().iter().all(|layer| layer.value().is_empty()));
}

#[test]
fn kv_cache_validates_layer_shapes() {
    let error = KvCache::new(
        vec![LayerCache::new(vec![0.0; 3], vec![0.0; 4])],
        1,
        1,
        1,
        4,
    )
    .expect_err("shape validation should fail");

    match error {
        Error::InvalidKvCache { reason } => {
            assert!(reason.contains("key length"));
        }
        other => panic!("expected invalid KV cache error, got {other:?}"),
    }
}

#[test]
fn logits_expose_shape_and_values() {
    let logits = Logits::new(vec![0.0, 1.0, 2.0, 3.0], 1, 2, 2).unwrap();

    assert_eq!(logits.shape(), (1, 2, 2));
    assert_eq!(logits.batch_size(), 1);
    assert_eq!(logits.sequence_length(), 2);
    assert_eq!(logits.vocab_size(), 2);
    assert_eq!(logits.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(logits.get(0, 1, 1), Some(3.0));
    assert_eq!(logits.get(1, 0, 0), None);
}

#[test]
fn rejects_invalid_logits_shapes() {
    let error = Logits::new(vec![0.0; 3], 1, 2, 2).expect_err("shape validation should fail");

    match error {
        Error::InvalidDecoderOutput { reason } => {
            assert!(reason.contains("does not match shape"));
        }
        other => panic!("expected invalid decoder output error, got {other:?}"),
    }
}

#[test]
fn official_decoder_runs_when_assets_are_available() {
    let Ok(model_directory) = std::env::var("LIGHTONOCR_DECODER_DIR") else {
        return;
    };

    let mut decoder = decoder_from_file(&model_directory).unwrap();
    let config = decoder.config().clone();
    let input_embeddings =
        InputEmbeddings::new(vec![0.0; config.hidden_size], 1, 1, config.hidden_size).unwrap();
    let attention_mask = AttentionMask::ones(1);
    let kv_cache = decoder.empty_kv_cache(1).unwrap();

    let output = decoder
        .decode(&input_embeddings, &attention_mask, &kv_cache)
        .unwrap();

    assert_eq!(output.logits.shape(), (1, 1, config.vocab_size));
    assert_eq!(
        output.kv_cache.shape(),
        (1, config.num_key_value_heads, 1, config.head_dim)
    );
}

#[test]
fn python_decoder_parity_when_enabled() {
    if std::env::var("LIGHTONOCR_RUN_PYTHON_DECODER_PARITY").is_err() {
        return;
    }

    let model_directory =
        std::env::var("LIGHTONOCR_DECODER_DIR").expect("LIGHTONOCR_DECODER_DIR is required");
    let mut decoder = decoder_from_file(&model_directory).unwrap();
    let config = decoder.config().clone();
    let input_embeddings =
        InputEmbeddings::new(vec![0.0; config.hidden_size], 1, 1, config.hidden_size).unwrap();
    let attention_mask = AttentionMask::ones(1);
    let kv_cache = decoder.empty_kv_cache(1).unwrap();
    let rust_output = decoder
        .decode(&input_embeddings, &attention_mask, &kv_cache)
        .unwrap();

    let script = format!(
        r#"
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

model_dir = Path({model_directory:?})
model_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"
hidden_size = {hidden_size}
num_layers = {num_layers}
num_key_value_heads = {num_key_value_heads}
head_dim = {head_dim}

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
input_names = {{input.name for input in session.get_inputs()}}
inputs = {{
    "inputs_embeds": np.zeros((1, 1, hidden_size), dtype=np.float32),
    "attention_mask": np.ones((1, 1), dtype=np.int64),
}}
if "use_cache_branch" in input_names:
    inputs["use_cache_branch"] = np.array(False)
for layer in range(num_layers):
    inputs[f"past_key_values.{{layer}}.key"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)
    inputs[f"past_key_values.{{layer}}.value"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)

output_values = session.run(None, inputs)
outputs = {{output.name: value for output, value in zip(session.get_outputs(), output_values)}}
payload = {{
    "logits_shape": list(outputs["logits"].shape),
    "logits_data": outputs["logits"].reshape(-1).tolist(),
    "cache_shapes": [],
    "cache_data": [],
}}
for layer in range(num_layers):
    for kind in ("key", "value"):
        value = outputs[f"present.{{layer}}.{{kind}}"]
        payload["cache_shapes"].append(list(value.shape))
        payload["cache_data"].append(value.reshape(-1).tolist())
print(json.dumps(payload))
"#,
        model_directory = model_directory,
        hidden_size = config.hidden_size,
        num_layers = config.num_hidden_layers,
        num_key_value_heads = config.num_key_value_heads,
        head_dim = config.head_dim
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

    let expected: PythonDecoderOutput =
        serde_json::from_slice(&output.stdout).expect("invalid python JSON output");
    assert_eq!(rust_output.logits.shape(), expected.logits_shape_tuple());
    assert_eq!(
        rust_output.logits.as_slice().len(),
        expected.logits_data.len()
    );
    for (actual, expected) in rust_output
        .logits
        .as_slice()
        .iter()
        .zip(expected.logits_data.iter())
    {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "logit mismatch: actual={actual}, expected={expected}"
        );
    }

    assert_eq!(expected.cache_shapes.len(), config.num_hidden_layers * 2);
    assert_eq!(expected.cache_data.len(), config.num_hidden_layers * 2);
    for (layer_index, layer) in rust_output.kv_cache.layers().iter().enumerate() {
        let key_index = layer_index * 2;
        let value_index = key_index + 1;
        assert_eq!(
            expected.cache_shape_tuple(key_index),
            rust_output.kv_cache.shape()
        );
        assert_eq!(
            expected.cache_shape_tuple(value_index),
            rust_output.kv_cache.shape()
        );
        assert_slice_close(layer.key(), &expected.cache_data[key_index]);
        assert_slice_close(layer.value(), &expected.cache_data[value_index]);
    }
}

fn assert_slice_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "cache mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug, Deserialize)]
struct PythonDecoderOutput {
    logits_shape: Vec<usize>,
    logits_data: Vec<f32>,
    cache_shapes: Vec<Vec<usize>>,
    cache_data: Vec<Vec<f32>>,
}

impl PythonDecoderOutput {
    fn logits_shape_tuple(&self) -> (usize, usize, usize) {
        assert_eq!(self.logits_shape.len(), 3);
        (
            self.logits_shape[0],
            self.logits_shape[1],
            self.logits_shape[2],
        )
    }

    fn cache_shape_tuple(&self, index: usize) -> (usize, usize, usize, usize) {
        let shape = &self.cache_shapes[index];
        assert_eq!(shape.len(), 4);
        (shape[0], shape[1], shape[2], shape[3])
    }
}
