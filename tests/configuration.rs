use std::path::{Path, PathBuf};

use fast_lightonocr::Error;
use fast_lightonocr::model::vision_encoder::VisionConfig;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn reports_missing_configuration_file() {
    let error =
        VisionConfig::from_file(fixture_path("lightonocr_missing_config").join("config.json"))
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
        VisionConfig::from_file(fixture_path("lightonocr_malformed_config").join("config.json"))
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
