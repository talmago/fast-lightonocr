//! High-level public API for LightOnOCR.

use std::path::{Path, PathBuf};

use ::image::DynamicImage;

use super::{
    Decoder, EmbeddingConfig, EmbeddingModel, FinishReason, GenerationConfig, GenerationOutput,
    VisionConfig, VisionEncoder,
};

use crate::model::{AttentionMask, ImageFeatures, InputEmbeddings};
use crate::processor::{Message, MessageContent, MessageRole};
use crate::profiling::{self, Stage};
use crate::util::ExecutionProvider;
use crate::{Error, Processor, Result};

/// Runtime configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeOptions {
    /// ONNX Runtime execution provider.
    pub execution_provider: ExecutionProvider,
}

/// Options used when loading a pretrained LightOnOCR model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightOnOCROptions {
    /// Runtime configuration.
    pub runtime: RuntimeOptions,

    /// Vision encoder ONNX filename relative to the model directory.
    pub vision_encoder: PathBuf,

    /// Token embedding ONNX filename relative to the model directory.
    pub embedding: PathBuf,

    /// Merged decoder ONNX filename relative to the model directory.
    pub decoder: PathBuf,

    /// Optional override for the maximum number of generated tokens.
    ///
    /// When omitted, the value loaded from `generation_config.json` is used.
    pub max_new_tokens: Option<usize>,
}

impl Default for LightOnOCROptions {
    fn default() -> Self {
        Self {
            runtime: RuntimeOptions::default(),
            vision_encoder: "onnx/vision_encoder.onnx".into(),
            embedding: "onnx/embed_tokens.onnx".into(),
            decoder: "onnx/decoder_model_merged.onnx".into(),
            max_new_tokens: None,
        }
    }
}

impl LightOnOCROptions {
    /// Returns options for the FP16 model files.
    #[must_use]
    pub fn fp16() -> Self {
        Self {
            vision_encoder: "onnx/vision_encoder_fp16.onnx".into(),
            embedding: "onnx/embed_tokens_fp16.onnx".into(),
            decoder: "onnx/decoder_model_merged_fp16.onnx".into(),
            ..Self::default()
        }
    }

    /// Returns options for the Q4 model files.
    #[must_use]
    pub fn q4() -> Self {
        Self {
            vision_encoder: "onnx/vision_encoder_q4.onnx".into(),
            embedding: "onnx/embed_tokens_q4.onnx".into(),
            decoder: "onnx/decoder_model_merged_q4.onnx".into(),
            ..Self::default()
        }
    }

    /// Returns a copy of these options with a custom generation limit.
    #[must_use]
    pub fn with_max_new_tokens(self, max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens: Some(max_new_tokens),
            ..self
        }
    }
}

/// OCR inference result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OCRResult {
    text: String,
    token_ids: Vec<i64>,
    finish_reason: FinishReason,
}

impl OCRResult {
    fn new(text: String, generated: GenerationOutput) -> Self {
        Self {
            text,
            token_ids: generated.token_ids().to_vec(),
            finish_reason: generated.finish_reason(),
        }
    }

    /// Returns the decoded OCR text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the generated token IDs.
    #[must_use]
    pub fn token_ids(&self) -> &[i64] {
        &self.token_ids
    }

    /// Returns the reason generation stopped.
    #[must_use]
    pub fn finish_reason(&self) -> FinishReason {
        self.finish_reason
    }

    /// Consumes the result and returns the decoded text.
    #[must_use]
    pub fn into_text(self) -> String {
        self.text
    }
}

/// High-level LightOnOCR pipeline.
///
/// The pipeline owns the processor and the three ONNX model runtimes:
///
/// - vision encoder;
/// - token embedding model;
/// - autoregressive decoder.
///
/// It orchestrates preprocessing, multimodal input preparation, token
/// generation, and final text decoding.
#[derive(Debug)]
pub struct LightOnOCR {
    processor: Processor,
    vision_encoder: VisionEncoder,
    embedding_model: EmbeddingModel,
    decoder: Decoder,
    runtime: RuntimeOptions,
}

impl LightOnOCR {
    /// Loads a pretrained model from a Hugging Face model directory.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        options: LightOnOCROptions,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let processor = Processor::from_dir(model_dir)?;

        let config_path = model_dir.join("config.json");

        let vision_encoder = VisionEncoder::from_model_path(
            model_dir.join(&options.vision_encoder),
            VisionConfig::from_file(&config_path)?,
        )?;

        let embedding_model = EmbeddingModel::from_model_path(
            model_dir.join(&options.embedding),
            EmbeddingConfig::from_file(&config_path)?,
        )?;

        let mut decoder = Decoder::from_model_path(model_dir.join(&options.decoder))?;

        if let Some(max_new_tokens) = options.max_new_tokens {
            decoder.generation_config_mut().max_new_tokens = max_new_tokens;
        }

        Ok(Self {
            processor,
            vision_encoder,
            embedding_model,
            decoder,
            runtime: options.runtime,
        })
    }

    /// Returns the processor.
    #[must_use]
    pub fn processor(&self) -> &Processor {
        &self.processor
    }

    /// Returns the runtime configuration.
    #[must_use]
    pub fn runtime(&self) -> RuntimeOptions {
        self.runtime
    }

    /// Returns the generation configuration.
    #[must_use]
    pub fn generation_config(&self) -> &GenerationConfig {
        self.decoder.generation_config()
    }

    /// Returns mutable access to the generation configuration.
    pub fn generation_config_mut(&mut self) -> &mut GenerationConfig {
        self.decoder.generation_config_mut()
    }

    /// Processes an in-memory image.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process(
        &mut self,
        image: &DynamicImage,
        system_prompt: Option<&str>,
    ) -> Result<OCRResult> {
        profiling::run(|| {
            let (input_embeddings, attention_mask) = self.prepare_inputs(image, system_prompt)?;

            let generated = self.decoder.generate(
                input_embeddings,
                attention_mask,
                &mut self.embedding_model,
            )?;

            self.decode_result(generated)
        })
    }

    /// Processes an in-memory image while streaming decoded text chunks.
    ///
    /// The callback receives decoded text deltas as generation progresses.
    /// Special tokens are omitted from streamed chunks and from the returned
    /// final result.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process_streaming(
        &mut self,
        image: &DynamicImage,
        system_prompt: Option<&str>,
        mut on_text: impl FnMut(&str),
    ) -> Result<OCRResult> {
        profiling::run(|| {
            let (input_embeddings, attention_mask) = self.prepare_inputs(image, system_prompt)?;

            let tokenizer = self.processor.tokenizer();
            let mut stream_token_ids = Vec::new();
            let mut streamed_text = String::new();
            let mut streaming_error = None;

            let generated = self.decoder.generate_streaming(
                input_embeddings,
                attention_mask,
                &mut self.embedding_model,
                |token_id| {
                    if streaming_error.is_some() {
                        return;
                    }

                    stream_token_ids.push(token_id);

                    let decoded = {
                        let _timer = profiling::start(Stage::StreamDecode);
                        match tokenizer.decode_skip_special_tokens(&stream_token_ids) {
                            Ok(decoded) => decoded,
                            Err(error) => {
                                streaming_error = Some(error);
                                return;
                            }
                        }
                    };

                    let chunk = decoded_delta(&streamed_text, &decoded);
                    if !chunk.is_empty() {
                        let _timer = profiling::start(Stage::StreamCallback);
                        on_text(chunk);
                    }

                    streamed_text = decoded;
                },
            )?;

            if let Some(error) = streaming_error {
                return Err(error);
            }

            self.decode_result(generated)
        })
    }

    /// Loads and processes an image from disk.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process_file(
        &mut self,
        image_path: impl AsRef<Path>,
        system_prompt: Option<&str>,
    ) -> Result<OCRResult> {
        let image = image::open(image_path)?;
        self.process(&image, system_prompt)
    }

    /// Loads and processes an image from disk while streaming decoded text chunks.
    ///
    /// The callback receives decoded text deltas as generation progresses.
    /// Special tokens are omitted from streamed chunks and from the returned
    /// final result.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process_file_streaming(
        &mut self,
        image_path: impl AsRef<Path>,
        system_prompt: Option<&str>,
        on_text: impl FnMut(&str),
    ) -> Result<OCRResult> {
        let image = image::open(image_path)?;
        self.process_streaming(&image, system_prompt, on_text)
    }

    /// Converts an image and optional prompt into decoder-ready inputs.
    ///
    /// This performs the complete multimodal preparation flow:
    ///
    /// 1. preprocess the image;
    /// 2. render and tokenize the conversation;
    /// 3. create the attention mask;
    /// 4. run the vision encoder;
    /// 5. embed the prompt tokens;
    /// 6. replace image-token embeddings with visual features.
    fn prepare_inputs(
        &mut self,
        image: &DynamicImage,
        system_prompt: Option<&str>,
    ) -> Result<(InputEmbeddings, AttentionMask)> {
        let _prepare_timer = profiling::start(Stage::PrepareInputs);
        let mut content = Vec::new();

        if let Some(prompt) = system_prompt {
            content.push(MessageContent::Text(prompt.to_owned()));
        }

        content.push(MessageContent::Image);

        let messages = vec![Message {
            role: MessageRole::User,
            content,
        }];

        let processed = self
            .processor
            .process(&messages, std::slice::from_ref(image))?;

        let image_features = self.vision_encoder.encode(&processed.pixel_values)?;

        let text_embeddings = {
            let _timer = profiling::start(Stage::PromptEmbedding);
            self.embedding_model.embed(&processed.input_ids)?
        };

        let input_embeddings = {
            let _timer = profiling::start(Stage::MergeImageFeatures);
            merge_image_features(
                &processed.input_ids,
                &text_embeddings,
                &image_features,
                self.decoder.image_token_id(),
            )?
        };

        Ok((input_embeddings, processed.attention_mask))
    }

    /// Decodes generated token IDs into the final OCR result.
    fn decode_result(&self, generated: GenerationOutput) -> Result<OCRResult> {
        let text = {
            let _timer = profiling::start(Stage::FinalTextDecode);
            self.processor
                .tokenizer()
                .decode_skip_special_tokens(generated.token_ids())?
        };

        Ok(OCRResult::new(text, generated))
    }
}

/// Replaces image-placeholder token embeddings with vision encoder features.
fn merge_image_features(
    token_ids: &[i64],
    input_embeddings: &InputEmbeddings,
    image_features: &ImageFeatures,
    image_token_id: i64,
) -> Result<InputEmbeddings> {
    let (embedding_batch_size, sequence_length, embedding_hidden_size) = input_embeddings.shape();

    let (feature_batch_size, feature_count, feature_hidden_size) = image_features.shape();

    if embedding_batch_size != 1 {
        return Err(Error::Inference {
            reason: format!("only batch size 1 is supported, found {embedding_batch_size}"),
        });
    }

    if feature_batch_size != 1 {
        return Err(Error::Inference {
            reason: format!(
                "only batch size 1 is supported for image features, \
                 found {feature_batch_size}"
            ),
        });
    }

    if token_ids.len() != sequence_length {
        return Err(Error::Inference {
            reason: format!(
                "token ID sequence length {} does not match embedding \
                 sequence length {sequence_length}",
                token_ids.len(),
            ),
        });
    }

    if embedding_hidden_size != feature_hidden_size {
        return Err(Error::Inference {
            reason: format!(
                "embedding hidden size {embedding_hidden_size} does not match \
                 image feature hidden size {feature_hidden_size}"
            ),
        });
    }

    let image_token_positions = token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token_id)| (token_id == image_token_id).then_some(index))
        .collect::<Vec<_>>();

    if image_token_positions.len() != feature_count {
        return Err(Error::Inference {
            reason: format!(
                "prompt contains {} image placeholder tokens but vision \
                 encoder produced {feature_count} image features",
                image_token_positions.len(),
            ),
        });
    }

    let mut merged = input_embeddings.as_slice().to_vec();
    let features = image_features.as_slice();

    for (feature_index, &sequence_index) in image_token_positions.iter().enumerate() {
        let embedding_start = sequence_index * embedding_hidden_size;
        let embedding_end = embedding_start + embedding_hidden_size;

        let feature_start = feature_index * feature_hidden_size;
        let feature_end = feature_start + feature_hidden_size;

        merged[embedding_start..embedding_end]
            .copy_from_slice(&features[feature_start..feature_end]);
    }

    InputEmbeddings::new(
        merged,
        embedding_batch_size,
        sequence_length,
        embedding_hidden_size,
    )
    .map_err(|error| Error::Inference {
        reason: error.to_string(),
    })
}

fn decoded_delta<'a>(previous: &str, current: &'a str) -> &'a str {
    current
        .strip_prefix(previous)
        .unwrap_or_else(|| &current[common_prefix_len(previous, current)..])
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    let mut left_chars = left.chars();
    let mut prefix_len = 0;

    for (right_index, right_char) in right.char_indices() {
        match left_chars.next() {
            Some(left_char) if left_char == right_char => {
                prefix_len = right_index + right_char.len_utf8();
            }
            _ => break,
        }
    }

    prefix_len
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_TOKEN_ID: i64 = 151_655;

    #[test]
    fn merge_image_features_replaces_image_token_embeddings_in_place() {
        let token_ids = [10, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID];
        let input_embeddings = InputEmbeddings::new(
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            1,
            4,
            2,
        )
        .unwrap();
        let image_features = ImageFeatures::new(vec![100.0, 101.0, 200.0, 201.0], 1, 2, 2).unwrap();

        let merged = merge_image_features(
            &token_ids,
            &input_embeddings,
            &image_features,
            IMAGE_TOKEN_ID,
        )
        .unwrap();

        assert_eq!(
            merged.as_slice(),
            &[10.0, 11.0, 100.0, 101.0, 30.0, 31.0, 200.0, 201.0]
        );
    }

    #[test]
    fn merge_image_features_rejects_placeholder_feature_count_mismatch() {
        let token_ids = [IMAGE_TOKEN_ID];
        let input_embeddings = InputEmbeddings::new(vec![0.0, 1.0], 1, 1, 2).unwrap();
        let image_features = ImageFeatures::new(vec![10.0, 11.0, 20.0, 21.0], 1, 2, 2).unwrap();

        let error = merge_image_features(
            &token_ids,
            &input_embeddings,
            &image_features,
            IMAGE_TOKEN_ID,
        )
        .expect_err("placeholder count mismatch should fail");

        match error {
            Error::Inference { reason } => {
                assert!(reason.contains("image placeholder tokens"));
            }
            other => panic!("expected inference error, got {other:?}"),
        }
    }

    #[test]
    fn merge_image_features_rejects_hidden_size_mismatch() {
        let token_ids = [IMAGE_TOKEN_ID];
        let input_embeddings = InputEmbeddings::new(vec![0.0, 1.0], 1, 1, 2).unwrap();
        let image_features = ImageFeatures::new(vec![10.0, 11.0, 12.0], 1, 1, 3).unwrap();

        let error = merge_image_features(
            &token_ids,
            &input_embeddings,
            &image_features,
            IMAGE_TOKEN_ID,
        )
        .expect_err("hidden size mismatch should fail");

        match error {
            Error::Inference { reason } => {
                assert!(reason.contains("hidden size"));
            }
            other => panic!("expected inference error, got {other:?}"),
        }
    }

    #[test]
    fn decoded_delta_returns_new_suffix() {
        assert_eq!(decoded_delta("total", "total due"), " due");
    }

    #[test]
    fn decoded_delta_falls_back_to_common_prefix() {
        assert_eq!(decoded_delta("hello ", "hello,"), ",");
    }
}
