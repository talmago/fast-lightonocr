//! Shared utilities and crate-wide error handling.

pub(crate) mod json;

use std::path::PathBuf;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Crate-wide error type.
///
/// Future milestones will add more specific variants for configuration,
/// tokenizer, image processing, runtime, tensor validation, and inference
/// failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A required Hugging Face configuration file was not present.
    #[error("missing configuration file: {}", path.display())]
    MissingConfigurationFile {
        /// Path to the missing configuration file.
        path: PathBuf,
    },

    /// A configuration file could not be read from disk.
    #[error("failed to read configuration file {}: {source}", path.display())]
    ReadConfigurationFile {
        /// Path to the configuration file that could not be read.
        path: PathBuf,

        /// Underlying file-system error.
        #[source]
        source: std::io::Error,
    },

    /// A configuration file contained malformed JSON syntax.
    #[error("malformed JSON in configuration file {}: {source}", path.display())]
    MalformedConfigurationJson {
        /// Path to the malformed configuration file.
        path: PathBuf,

        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },

    /// A configuration file was syntactically valid but did not match the expected schema.
    #[error("invalid configuration in {}: {source}", path.display())]
    InvalidConfiguration {
        /// Path to the invalid configuration file.
        path: PathBuf,

        /// JSON deserialization error.
        #[source]
        source: serde_json::Error,
    },

    /// The configuration describes a model variant this crate does not support yet.
    #[error("unsupported configuration: {reason}")]
    UnsupportedConfiguration {
        /// Explanation of the unsupported configuration.
        reason: String,
    },

    /// A required image processor asset was not present in the model directory.
    #[error("missing processor asset: {}", path.display())]
    MissingProcessorAsset {
        /// Path to the missing processor asset.
        path: PathBuf,
    },

    /// An image processor asset could not be read from disk.
    #[error("failed to read processor asset {}: {source}", path.display())]
    ReadProcessorAsset {
        /// Path to the processor asset that could not be read.
        path: PathBuf,

        /// Underlying file-system error.
        #[source]
        source: std::io::Error,
    },

    /// A processor JSON asset contained malformed JSON syntax.
    #[error("malformed JSON in processor asset {}: {source}", path.display())]
    MalformedProcessorJson {
        /// Path to the malformed processor asset.
        path: PathBuf,

        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },

    /// A processor JSON asset was syntactically valid but did not match the expected schema.
    #[error("invalid processor asset {}: {source}", path.display())]
    InvalidProcessorJson {
        /// Path to the invalid processor asset.
        path: PathBuf,

        /// JSON deserialization error.
        #[source]
        source: serde_json::Error,
    },

    /// The image processor configuration describes unsupported preprocessing.
    #[error("unsupported processor configuration: {reason}")]
    UnsupportedProcessorConfiguration {
        /// Explanation of the unsupported processor configuration.
        reason: String,
    },

    /// An input image could not be loaded.
    #[error("failed to load image {}: {source}", path.display())]
    ImageLoad {
        /// Path to the image that could not be loaded.
        path: PathBuf,

        /// Underlying image loading error.
        #[source]
        source: image::ImageError,
    },

    /// Image preprocessing failed.
    #[error("image processing failed: {reason}")]
    ImageProcessing {
        /// Explanation of the image processing failure.
        reason: String,
    },

    /// A required tokenizer asset was not present in the model directory.
    #[error("missing tokenizer asset: {}", path.display())]
    MissingTokenizerAsset {
        /// Path to the missing tokenizer asset.
        path: PathBuf,
    },

    /// A tokenizer asset could not be read from disk.
    #[error("failed to read tokenizer asset {}: {source}", path.display())]
    ReadTokenizerAsset {
        /// Path to the tokenizer asset that could not be read.
        path: PathBuf,

        /// Underlying file-system error.
        #[source]
        source: std::io::Error,
    },

    /// A tokenizer JSON asset contained malformed JSON syntax.
    #[error("malformed JSON in tokenizer asset {}: {source}", path.display())]
    MalformedTokenizerJson {
        /// Path to the malformed tokenizer asset.
        path: PathBuf,

        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },

    /// A tokenizer JSON asset was syntactically valid but did not match the expected schema.
    #[error("invalid tokenizer asset {}: {source}", path.display())]
    InvalidTokenizerJson {
        /// Path to the invalid tokenizer asset.
        path: PathBuf,

        /// JSON deserialization error.
        #[source]
        source: serde_json::Error,
    },

    /// The tokenizer backend failed to initialize from `tokenizer.json`.
    #[error("failed to initialize tokenizer from {}: {source}", path.display())]
    TokenizerInitialization {
        /// Path to `tokenizer.json`.
        path: PathBuf,

        /// Underlying tokenizer library error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Text encoding failed in the tokenizer backend.
    #[error("tokenizer encoding failed: {source}")]
    TokenizerEncoding {
        /// Underlying tokenizer library error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Token ID decoding failed in the tokenizer backend.
    #[error("tokenizer decoding failed: {source}")]
    TokenizerDecoding {
        /// Underlying tokenizer library error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A model-ready token ID cannot be represented by the tokenizer backend.
    #[error("invalid token ID for tokenizer backend: {token_id}")]
    InvalidTokenId {
        /// Invalid token identifier.
        token_id: i64,
    },

    /// A configured special token could not be resolved to a token ID.
    #[error("configured special token {name}={token:?} is missing from tokenizer.json")]
    MissingSpecialTokenId {
        /// Special-token field name.
        name: &'static str,

        /// Special-token string content.
        token: String,
    },

    /// Autoregressive inference failed before or during generation.
    #[error("inference failed: {reason}")]
    Inference {
        /// Explanation of the inference failure.
        reason: String,
    },

    /// A required decoder ONNX model was not present.
    #[error("missing decoder model: {}", path.display())]
    MissingDecoderModel {
        /// Path to the missing decoder model.
        path: PathBuf,
    },

    /// The decoder ONNX model did not match the documented contract.
    #[error("invalid decoder model: {reason}")]
    InvalidDecoderModel {
        /// Explanation of the contract mismatch.
        reason: String,
    },

    /// The decoder ONNX session could not be loaded.
    #[error("failed to load decoder model {}: {source}", path.display())]
    DecoderModelLoad {
        /// Path to the decoder model.
        path: PathBuf,

        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Decoder inputs did not satisfy the decoder contract.
    #[error("invalid decoder input: {reason}")]
    InvalidDecoderInput {
        /// Explanation of the invalid input.
        reason: String,
    },

    /// A tensor could not be created for the decoder.
    #[error("failed to create decoder tensor: {source}")]
    DecoderTensorCreation {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Decoder inference failed.
    #[error("decoder inference failed: {source}")]
    DecoderInference {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Decoder output did not satisfy the documented contract.
    #[error("invalid decoder output: {reason}")]
    InvalidDecoderOutput {
        /// Explanation of the invalid output.
        reason: String,
    },

    /// A decoder key/value cache was malformed.
    #[error("invalid KV cache: {reason}")]
    InvalidKvCache {
        /// Explanation of the malformed cache.
        reason: String,
    },

    /// A required embedding ONNX model was not present.
    #[error("missing embedding model: {}", path.display())]
    MissingEmbeddingModel {
        /// Path to the missing embedding model.
        path: PathBuf,
    },

    /// The embedding ONNX model did not match the documented contract.
    #[error("invalid embedding model: {reason}")]
    InvalidEmbeddingModel {
        /// Explanation of the contract mismatch.
        reason: String,
    },

    /// The embedding ONNX session could not be loaded.
    #[error("failed to load embedding model {}: {source}", path.display())]
    EmbeddingModelLoad {
        /// Path to the embedding model.
        path: PathBuf,

        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Token IDs did not satisfy the embedding input contract.
    #[error("invalid embedding input: {reason}")]
    InvalidEmbeddingInput {
        /// Explanation of the invalid input.
        reason: String,
    },

    /// A tensor could not be created for the embedding model.
    #[error("failed to create embedding tensor: {source}")]
    EmbeddingTensorCreation {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Embedding model inference failed.
    #[error("embedding inference failed: {source}")]
    EmbeddingInference {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Embedding model output did not satisfy the documented contract.
    #[error("invalid embedding output: {reason}")]
    InvalidEmbeddingOutput {
        /// Explanation of the invalid output.
        reason: String,
    },

    /// A required vision encoder ONNX model was not present.
    #[error("missing vision model: {}", path.display())]
    MissingVisionModel {
        /// Path to the missing vision model.
        path: PathBuf,
    },

    /// The vision encoder ONNX model did not match the documented contract.
    #[error("invalid vision model: {reason}")]
    InvalidVisionModel {
        /// Explanation of the contract mismatch.
        reason: String,
    },

    /// The vision encoder ONNX session could not be loaded.
    #[error("failed to load vision model {}: {source}", path.display())]
    VisionModelLoad {
        /// Path to the vision model.
        path: PathBuf,

        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Pixel values did not satisfy the vision encoder input contract.
    #[error("invalid vision input: {reason}")]
    InvalidVisionInput {
        /// Explanation of the invalid input.
        reason: String,
    },

    /// A tensor could not be created for the vision encoder.
    #[error("failed to create vision tensor: {source}")]
    VisionTensorCreation {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Vision encoder inference failed.
    #[error("vision encoder inference failed: {source}")]
    VisionInference {
        /// Underlying ONNX Runtime error.
        #[source]
        source: ort::Error,
    },

    /// Vision encoder output did not satisfy the documented contract.
    #[error("invalid vision output: {reason}")]
    InvalidVisionOutput {
        /// Explanation of the invalid output.
        reason: String,
    },

    /// Placeholder error for functionality that has not been implemented yet.
    #[error("{feature} is not implemented yet")]
    NotImplemented {
        /// Name of the unavailable feature.
        feature: &'static str,
    },
}

impl From<image::ImageError> for Error {
    fn from(error: image::ImageError) -> Self {
        Self::ImageProcessing {
            reason: error.to_string(),
        }
    }
}
