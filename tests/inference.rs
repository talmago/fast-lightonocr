use std::path::{Path, PathBuf};

use fast_lightonocr::model::decoder::GenerationConfig;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn loads_generation_config_from_model_directory() {
    let config = GenerationConfig::from_file(
        fixture_path("lightonocr_config").join("generation_config.json"),
    )
    .unwrap();

    assert_eq!(config.bos_token_id, 151643);
    assert_eq!(config.eos_token_ids, vec![151645, 151643]);
    assert_eq!(config.pad_token_id, 151643);
    assert_eq!(config.temperature, 0.2);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.top_p, 0.9);
    assert!(config.do_sample);
    assert!(!config.trust_remote_code);
}

#[test]
fn generation_config_default_bounds_generation() {
    let config = GenerationConfig::from_file(
        fixture_path("lightonocr_config").join("generation_config.json"),
    )
    .unwrap();

    assert_eq!(config.max_new_tokens, 512);
}
