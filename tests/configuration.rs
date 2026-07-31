use std::path::{Path, PathBuf};

use fast_lightonocr::Error;
use fast_lightonocr::model::config::{Activation, DataType, ModelConfig, ModelType};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn loads_top_level_model_config() {
    let configuration =
        ModelConfig::from_file(fixture_path("lightonocr_config").join("config.json")).unwrap();

    assert_eq!(configuration.model_type, ModelType::LightOnOcr);
    assert_eq!(configuration.dtype, DataType::BFloat16);
    assert_eq!(configuration.image_token_index, 151655);
    assert_eq!(configuration.projector_hidden_act, Activation::Gelu);
    assert_eq!(configuration.spatial_merge_size, 2);
    assert_eq!(configuration.vision_feature_layer, -1);
}

#[test]
fn reports_missing_configuration_file() {
    let error =
        ModelConfig::from_file(fixture_path("lightonocr_missing_config").join("config.json"))
            .expect_err("configuration loading should fail");

    match error {
        Error::MissingConfigurationFile { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("config.json")
            );
        }
        other => panic!("expected missing configuration error, got {other:?}"),
    }
}

#[test]
fn reports_malformed_json() {
    let error =
        ModelConfig::from_file(fixture_path("lightonocr_malformed_config").join("config.json"))
            .expect_err("configuration loading should fail");

    match error {
        Error::MalformedConfigurationJson { path, .. } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("config.json")
            );
        }
        other => panic!("expected malformed JSON error, got {other:?}"),
    }
}
