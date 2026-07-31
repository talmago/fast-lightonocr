use std::path::{Path, PathBuf};

use fast_lightonocr::tokenizer::{PaddingSide, Tokenizer};
use fast_lightonocr::{Error, Result};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn tokenizer_from_directory(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let directory = path.as_ref();
    let special_tokens_map = directory.join("special_tokens_map.json");

    Tokenizer::from_files(
        directory.join("tokenizer.json"),
        directory.join("tokenizer_config.json"),
        Some(&special_tokens_map),
    )
}

#[test]
fn loads_tokenizer_assets_and_resolves_special_tokens() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    assert_eq!(tokenizer.config().padding_side, PaddingSide::Right);
    assert!(tokenizer.special_tokens_map().is_some());

    let special_tokens = tokenizer.special_token_ids();

    assert_eq!(special_tokens.bos_token_id, Some(0));
    assert_eq!(special_tokens.eos_token_id, 1);
    assert_eq!(special_tokens.pad_token_id, 2);
    assert_eq!(special_tokens.unk_token_id, Some(3));
}

#[test]
fn encodes_text_to_token_ids() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    let ids = tokenizer.encode("hello world").unwrap();

    assert_eq!(ids, [4, 5]);
}

#[test]
fn resolves_token_ids() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    assert_eq!(tokenizer.token_to_id("<s>").unwrap(), 0);
    assert_eq!(tokenizer.token_to_id("</s>").unwrap(), 1);
    assert_eq!(tokenizer.token_to_id("<pad>").unwrap(), 2);
    assert_eq!(tokenizer.token_to_id("<unk>").unwrap(), 3);
    assert_eq!(tokenizer.token_to_id("hello").unwrap(), 4);
    assert_eq!(tokenizer.token_to_id("world").unwrap(), 5);
}

#[test]
fn resolves_tokens_from_ids() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    assert_eq!(tokenizer.id_to_token(0).as_deref(), Some("<s>"));
    assert_eq!(tokenizer.id_to_token(1).as_deref(), Some("</s>"));
    assert_eq!(tokenizer.id_to_token(2).as_deref(), Some("<pad>"));
    assert_eq!(tokenizer.id_to_token(3).as_deref(), Some("<unk>"));
    assert_eq!(tokenizer.id_to_token(4).as_deref(), Some("hello"));
    assert_eq!(tokenizer.id_to_token(5).as_deref(), Some("world"));
    assert_eq!(tokenizer.id_to_token(9999), None);
}

#[test]
fn matches_hugging_face_expected_special_tokens() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    let ids = tokenizer.encode("<s> hello </s>").unwrap();

    assert_eq!(ids, [0, 4, 1]);
    assert_eq!(tokenizer.decode(&ids).unwrap(), "<s> hello </s>");
    assert_eq!(tokenizer.decode_skip_special_tokens(&ids).unwrap(), "hello");
}

#[test]
fn decodes_batches() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    let batch = [vec![4, 5], vec![6, 7], vec![0, 4, 1]];

    let decoded = tokenizer.decode_batch(&batch).unwrap();
    let decoded_without_special_tokens =
        tokenizer.decode_batch_skip_special_tokens(&batch).unwrap();

    assert_eq!(decoded, ["hello world", "rust OCR", "<s> hello </s>"]);

    assert_eq!(
        decoded_without_special_tokens,
        ["hello world", "rust OCR", "hello"]
    );
}

#[test]
fn encode_decode_round_trip_is_deterministic() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    let ids = tokenizer.encode("rust OCR").unwrap();
    let decoded = tokenizer.decode(&ids).unwrap();
    let reencoded = tokenizer.encode(&decoded).unwrap();

    assert_eq!(decoded, "rust OCR");
    assert_eq!(ids, reencoded);
}

#[test]
fn reports_missing_tokenizer_json() {
    let error = tokenizer_from_directory(fixture_path("hf_tokenizer_missing_tokenizer_json"))
        .expect_err("tokenizer loading should fail");

    match error {
        Error::MissingTokenizerAsset { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("tokenizer.json")
            );
        }
        other => panic!("expected missing tokenizer asset error, got {other:?}"),
    }
}

#[test]
fn rejects_negative_token_ids_for_decoding() {
    let tokenizer = tokenizer_from_directory(fixture_path("hf_tokenizer")).unwrap();

    let error = tokenizer
        .decode(&[-1])
        .expect_err("negative token IDs cannot be decoded");

    match error {
        Error::InvalidTokenId { token_id } => assert_eq!(token_id, -1),
        other => panic!("expected invalid token ID error, got {other:?}"),
    }
}

#[test]
fn official_lightonocr_tokenizer_parity_when_assets_are_available() {
    let Ok(model_directory) = std::env::var("LIGHTONOCR_TOKENIZER_DIR") else {
        return;
    };

    let tokenizer = tokenizer_from_directory(model_directory).unwrap();

    let special_tokens = tokenizer.special_token_ids();

    assert_eq!(special_tokens.bos_token_id, None);
    assert_eq!(special_tokens.eos_token_id, 151645);
    assert_eq!(special_tokens.pad_token_id, 151643);

    assert_eq!(tokenizer.decode(&[151645]).unwrap(), "<|im_end|>");
    assert_eq!(tokenizer.decode(&[151643]).unwrap(), "<|endoftext|>");
}
