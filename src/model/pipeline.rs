//! High-level public API for LightOnOCR.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ::image::DynamicImage;

use super::decoder::{
    GenerationConfig, GenerationOutput, PixtralDecoder, QwenDecoder, QwenPromptState,
};
use super::vision_encoder::{PixtralVisionEncoder, QwenVisionEncoder};
use super::{
    AttentionMask, EmbeddingConfig, EmbeddingModel, FinishReason, ImageFeatures, InputEmbeddings,
    VisionConfig,
};
use crate::processor::{Message, MessageContent, MessageRole, PixtralProcessor, QwenProcessor};
use crate::tokenizer::Tokenizer;
use crate::util::RuntimeOptions;
use crate::{Error, Result};

const QWEN_PREFILL_GRAPH: &str = "onnx/decoder_prefill.onnx";
const QWEN_DECODE_GRAPH: &str = "onnx/decoder_decode.onnx";

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
    ///
    /// Used by the Pixtral stack. Qwen loads `onnx/decoder_prefill.onnx` and
    /// `onnx/decoder_decode.onnx` instead of this path.
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

    /// Returns a copy of these options with custom runtime configuration.
    #[must_use]
    pub fn with_runtime(self, runtime: RuntimeOptions) -> Self {
        Self { runtime, ..self }
    }
}

/// Wall-clock stage timings for one OCR request, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    /// Image load from disk when using [`LightOnOCR::process_file_with_timings`].
    pub io_ms: f64,
    /// Image preprocessing, prompt rendering, tokenization, and mask creation.
    pub prep_ms: f64,
    /// Vision encoder ONNX execution (including host output extraction).
    pub vision_ms: f64,
    /// Token embedding ONNX execution.
    pub embed_ms: f64,
    /// Image-feature merge into the prompt embedding sequence.
    pub merge_ms: f64,
    /// Autoregressive decoder generation and final text decoding.
    pub decode_ms: f64,
}

impl StageTimings {
    /// Returns the sum of all recorded stage durations.
    #[must_use]
    pub fn total_ms(&self) -> f64 {
        self.io_ms + self.prep_ms + self.vision_ms + self.embed_ms + self.merge_ms + self.decode_ms
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

/// Inference stacks implemented by this crate.
///
/// - [`Self::Pixtral`] is LightOnOCR-2-1B and LightOnOCR-3-1B: Pixtral vision,
///   Qwen3 full-attention decoder, merged KV-cache graph.
/// - [`Self::Qwen`] is LightOnOCR-3-0.8B (and later 4B): Qwen3.5 packed-patch
///   vision and a hybrid linear/full-attention decoder with separate prefill
///   and decode graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceArchitecture {
    /// Pixtral vision + Qwen3 full-attention decoder.
    Pixtral,
    /// Qwen3.5 vision-language stack.
    Qwen,
}

impl InferenceArchitecture {
    /// Classifies a parsed checkpoint `config.json`.
    ///
    /// The top-level `model_type` identifies the wrapper; nested
    /// `vision_config.model_type` and `text_config.model_type` select the
    /// inference stack. Unknown combinations return
    /// [`Error::UnsupportedConfiguration`] with the observed values.
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Self::from_config_at(config, None)
    }

    /// Classifies `config.json` and includes `path` in error messages when set.
    pub fn from_config_at(config: &serde_json::Value, path: Option<&Path>) -> Result<Self> {
        let model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let architectures = config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let normalized_architectures = architectures
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(|value| value.to_ascii_lowercase().replace('_', ""))
            .collect::<Vec<_>>();
        let text_model_type = config
            .pointer("/text_config/model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let vision_model_type = config
            .pointer("/vision_config/model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let layer_types = config
            .pointer("/text_config/layer_types")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let layer_type_names = layer_types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>();

        let location = path
            .map(|path| format!(" in {}", path.display()))
            .unwrap_or_default();

        if matches!(model_type, "lighton_ocr" | "mistral3")
            && normalized_architectures
                .iter()
                .any(|value| value == "lightonocrforconditionalgeneration")
            && text_model_type == "qwen3"
            && vision_model_type == "pixtral"
        {
            if layer_type_names
                .iter()
                .any(|layer_type| *layer_type != "full_attention")
            {
                return Err(Error::UnsupportedConfiguration {
                    reason: format!(
                        "Pixtral checkpoints must use full-attention decoder layers{location}; \
                         found layer_types={layer_type_names:?}"
                    ),
                });
            }
            return Ok(Self::Pixtral);
        }

        if model_type == "qwen3_5"
            && normalized_architectures
                .iter()
                .any(|value| value == "qwen35forconditionalgeneration")
            && text_model_type == "qwen3_5_text"
            && vision_model_type == "qwen3_5"
        {
            if layer_type_names.is_empty()
                || layer_type_names.iter().any(|layer_type| {
                    *layer_type != "linear_attention" && *layer_type != "full_attention"
                })
            {
                return Err(Error::UnsupportedConfiguration {
                    reason: format!(
                        "Qwen checkpoints must declare linear/full attention layers{location}; \
                         found layer_types={layer_type_names:?}"
                    ),
                });
            }
            return Ok(Self::Qwen);
        }

        Err(Error::UnsupportedConfiguration {
            reason: format!(
                "unsupported architecture{location}: model_type={model_type:?}, \
                 architectures={architectures:?}, text={text_model_type:?}, \
                 vision={vision_model_type:?}"
            ),
        })
    }
}

/// Decoder-ready tensors produced by [`InferenceBackend::prepare`].
struct PreparedInputs {
    embeddings: InputEmbeddings,
    attention_mask: AttentionMask,
    qwen: Option<QwenPromptState>,
}

/// Pixtral or Qwen inference stack selected from `config.json`.
///
/// Each variant owns that family's processor, vision encoder, and decoder.
/// Token embeddings stay on [`LightOnOCR`] because both families share the
/// same `embed_tokens` graph.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum InferenceBackend {
    /// LightOnOCR-2 / LightOnOCR-3-1B: NCHW Pixtral vision and merged KV decoder.
    Pixtral {
        /// Pixtral chat/image processor.
        processor: PixtralProcessor,
        /// Pixtral vision encoder session.
        vision: PixtralVisionEncoder,
        /// Full-attention decoder with host or CUDA KV cache.
        decoder: PixtralDecoder,
    },
    /// LightOnOCR-3-0.8B (and later 4B): packed-patch vision and hybrid decoder.
    Qwen {
        /// Qwen packed-patch processor.
        processor: QwenProcessor,
        /// Qwen vision encoder session.
        vision: QwenVisionEncoder,
        /// Prefill/decode graphs with linear-attention and KV state.
        decoder: QwenDecoder,
    },
}

impl InferenceBackend {
    fn load(
        model_dir: &Path,
        config: &serde_json::Value,
        architecture: InferenceArchitecture,
        options: &LightOnOCROptions,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        match architecture {
            InferenceArchitecture::Pixtral => {
                let processor = PixtralProcessor::from_dir(model_dir)?;
                let vision = PixtralVisionEncoder::from_model_path_with_runtime(
                    model_dir.join(&options.vision_encoder),
                    VisionConfig::from_file(&config_path)?,
                    &options.runtime,
                )?;
                let decoder = PixtralDecoder::from_model_path_with_runtime(
                    model_dir.join(&options.decoder),
                    &options.runtime,
                )?;
                Ok(Self::Pixtral {
                    processor,
                    vision,
                    decoder,
                })
            }
            InferenceArchitecture::Qwen => {
                let hidden_size = config
                    .pointer("/vision_config/out_hidden_size")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| Error::UnsupportedConfiguration {
                        reason: "Qwen vision_config.out_hidden_size is missing or invalid"
                            .to_owned(),
                    })?;
                let processor = QwenProcessor::from_dir(model_dir)?;
                let vision = QwenVisionEncoder::from_model_path(
                    model_dir.join(&options.vision_encoder),
                    hidden_size,
                    &options.runtime,
                )?;
                let decoder = QwenDecoder::from_model_path(
                    model_dir.join(QWEN_PREFILL_GRAPH),
                    model_dir.join(QWEN_DECODE_GRAPH),
                    &config_path,
                    model_dir.join("generation_config.json"),
                    &options.runtime,
                )?;
                Ok(Self::Qwen {
                    processor,
                    vision,
                    decoder,
                })
            }
        }
    }

    /// Returns the tokenizer used to decode generated tokens.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Pixtral { processor, .. } => processor.tokenizer(),
            Self::Qwen { processor, .. } => processor.tokenizer(),
        }
    }

    /// Returns the generation configuration for the active decoder.
    #[must_use]
    pub fn generation_config(&self) -> &GenerationConfig {
        match self {
            Self::Pixtral { decoder, .. } => decoder.generation_config(),
            Self::Qwen { decoder, .. } => decoder.generation_config(),
        }
    }

    /// Returns mutable access to the generation configuration.
    pub fn generation_config_mut(&mut self) -> &mut GenerationConfig {
        match self {
            Self::Pixtral { decoder, .. } => decoder.generation_config_mut(),
            Self::Qwen { decoder, .. } => decoder.generation_config_mut(),
        }
    }

    fn prepare(
        &mut self,
        image: &DynamicImage,
        system_prompt: Option<&str>,
        embedding_model: &mut EmbeddingModel,
    ) -> Result<(PreparedInputs, StageTimings)> {
        let mut timings = StageTimings::default();
        match self {
            Self::Pixtral {
                processor,
                vision,
                decoder,
            } => {
                let mut content = Vec::new();
                if let Some(prompt) = system_prompt {
                    content.push(MessageContent::Text(prompt.to_owned()));
                }
                content.push(MessageContent::Image);
                let messages = vec![Message {
                    role: MessageRole::User,
                    content,
                }];

                let prep_started = Instant::now();
                let processed = processor.process(&messages, std::slice::from_ref(image))?;
                timings.prep_ms = prep_started.elapsed().as_secs_f64() * 1000.0;

                let vision_started = Instant::now();
                let image_features = vision.encode_reuse(&processed.pixel_values)?;
                timings.vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;

                let embed_started = Instant::now();
                let text_embeddings = embedding_model.embed(&processed.input_ids)?;
                timings.embed_ms = embed_started.elapsed().as_secs_f64() * 1000.0;

                let merge_started = Instant::now();
                let embeddings = merge_image_features(
                    &processed.input_ids,
                    &text_embeddings,
                    image_features,
                    decoder.image_token_id(),
                )?;
                timings.merge_ms = merge_started.elapsed().as_secs_f64() * 1000.0;

                Ok((
                    PreparedInputs {
                        embeddings,
                        attention_mask: processed.attention_mask,
                        qwen: None,
                    },
                    timings,
                ))
            }
            Self::Qwen {
                processor,
                vision,
                decoder: _,
            } => {
                let prep_started = Instant::now();
                let processed = processor.process(image, system_prompt)?;
                timings.prep_ms = prep_started.elapsed().as_secs_f64() * 1000.0;

                let vision_started = Instant::now();
                let image_features = vision.encode(&processed)?;
                timings.vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;

                let embed_started = Instant::now();
                let text_embeddings = embedding_model.embed(&processed.input_ids)?;
                timings.embed_ms = embed_started.elapsed().as_secs_f64() * 1000.0;

                let merge_started = Instant::now();
                let embeddings = merge_image_features(
                    &processed.input_ids,
                    &text_embeddings,
                    &image_features,
                    processed.image_token_id,
                )?;
                timings.merge_ms = merge_started.elapsed().as_secs_f64() * 1000.0;

                Ok((
                    PreparedInputs {
                        embeddings,
                        attention_mask: processed.attention_mask,
                        qwen: Some(QwenPromptState::new(
                            processed.position_ids,
                            processed.rope_delta,
                        )),
                    },
                    timings,
                ))
            }
        }
    }

    fn generate(
        &mut self,
        prepared: PreparedInputs,
        embedding_model: &mut EmbeddingModel,
        on_token: impl FnMut(i64),
    ) -> Result<GenerationOutput> {
        match self {
            Self::Pixtral { decoder, .. } => decoder.generate_streaming(
                prepared.embeddings,
                prepared.attention_mask,
                embedding_model,
                on_token,
            ),
            Self::Qwen { decoder, .. } => {
                let state = prepared.qwen.ok_or_else(|| Error::Inference {
                    reason: "Qwen decoder requires multimodal position IDs".to_owned(),
                })?;
                decoder.generate(
                    prepared.embeddings,
                    prepared.attention_mask,
                    &state,
                    embedding_model,
                    on_token,
                )
            }
        }
    }
}

/// High-level LightOnOCR pipeline.
///
/// Architecture selection is automatic from `config.json`. OCR callers use
/// [`Self::from_pretrained`] and [`Self::process`]; family-specific sessions
/// are available through [`Self::backend`].
///
/// The optional `system_prompt` argument is kept for API compatibility. Pixtral
/// renders it as user-turn text beside the image. Qwen places it on the user
/// turn as well (for example `grounding`).
#[derive(Debug)]
pub struct LightOnOCR {
    backend: InferenceBackend,
    embedding_model: EmbeddingModel,
    runtime: RuntimeOptions,
}

impl LightOnOCR {
    /// Loads a pretrained model from a Hugging Face model directory.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        options: LightOnOCROptions,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config: serde_json::Value = crate::util::json::load_json_file(&config_path)?;
        let architecture = InferenceArchitecture::from_config_at(&config, Some(&config_path))?;
        let mut backend = InferenceBackend::load(model_dir, &config, architecture, &options)?;
        let embedding_model = EmbeddingModel::from_model_path_with_runtime(
            model_dir.join(&options.embedding),
            EmbeddingConfig::from_file(&config_path)?,
            &options.runtime,
        )?;
        if let Some(max_new_tokens) = options.max_new_tokens {
            backend.generation_config_mut().max_new_tokens = max_new_tokens;
        }
        Ok(Self {
            backend,
            embedding_model,
            runtime: options.runtime,
        })
    }

    /// Returns the family-specific inference stack.
    #[must_use]
    pub fn backend(&self) -> &InferenceBackend {
        &self.backend
    }

    /// Returns mutable access to the family-specific inference stack.
    pub fn backend_mut(&mut self) -> &mut InferenceBackend {
        &mut self.backend
    }

    /// Returns the runtime configuration.
    #[must_use]
    pub fn runtime(&self) -> RuntimeOptions {
        self.runtime
    }

    /// Returns the generation configuration.
    #[must_use]
    pub fn generation_config(&self) -> &GenerationConfig {
        self.backend.generation_config()
    }

    /// Returns mutable access to the generation configuration.
    pub fn generation_config_mut(&mut self) -> &mut GenerationConfig {
        self.backend.generation_config_mut()
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
        Ok(self.process_with_timings(image, system_prompt)?.0)
    }

    /// Processes an in-memory image and returns stage-level wall-clock timings.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process_with_timings(
        &mut self,
        image: &DynamicImage,
        system_prompt: Option<&str>,
    ) -> Result<(OCRResult, StageTimings)> {
        let (prepared, mut timings) =
            self.backend
                .prepare(image, system_prompt, &mut self.embedding_model)?;

        let decode_started = Instant::now();
        let generated = self
            .backend
            .generate(prepared, &mut self.embedding_model, |_| {})?;
        let result = self.decode_result(generated)?;
        timings.decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        Ok((result, timings))
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
        let (prepared, _timings) =
            self.backend
                .prepare(image, system_prompt, &mut self.embedding_model)?;

        let tokenizer = self.backend.tokenizer().clone();
        let mut stream_token_ids = Vec::new();
        let mut streamed_text = String::new();
        let mut streaming_error = None;

        let generated = self
            .backend
            .generate(prepared, &mut self.embedding_model, |token_id| {
                if streaming_error.is_some() {
                    return;
                }
                stream_token_ids.push(token_id);
                let decoded = match tokenizer.decode_skip_special_tokens(&stream_token_ids) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        streaming_error = Some(error);
                        return;
                    }
                };
                let chunk = decoded_delta(&streamed_text, &decoded);
                if !chunk.is_empty() {
                    on_text(chunk);
                }
                streamed_text = decoded;
            })?;

        if let Some(error) = streaming_error {
            return Err(error);
        }
        self.decode_result(generated)
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
        Ok(self.process_file_with_timings(image_path, system_prompt)?.0)
    }

    /// Loads and processes an image from disk, returning stage-level timings.
    ///
    /// When `system_prompt` is `None`, the processor uses the model's default
    /// prompt behavior.
    pub fn process_file_with_timings(
        &mut self,
        image_path: impl AsRef<Path>,
        system_prompt: Option<&str>,
    ) -> Result<(OCRResult, StageTimings)> {
        let io_started = Instant::now();
        let image = image::open(image_path)?;
        let mut timings = StageTimings {
            io_ms: io_started.elapsed().as_secs_f64() * 1000.0,
            ..StageTimings::default()
        };
        let (result, prep_timings) = self.process_with_timings(&image, system_prompt)?;
        timings.prep_ms = prep_timings.prep_ms;
        timings.vision_ms = prep_timings.vision_ms;
        timings.embed_ms = prep_timings.embed_ms;
        timings.merge_ms = prep_timings.merge_ms;
        timings.decode_ms = prep_timings.decode_ms;
        Ok((result, timings))
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

    fn decode_result(&self, generated: GenerationOutput) -> Result<OCRResult> {
        let text = self
            .backend
            .tokenizer()
            .decode_skip_special_tokens(generated.token_ids())?;
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
    use serde_json::json;

    const IMAGE_TOKEN_ID: i64 = 151_655;

    #[test]
    fn classifies_lightonocr2_pixtral_wrapper() {
        let config = json!({
            "model_type": "lighton_ocr",
            "architectures": ["LightOnOcrForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3",
                "layer_types": ["full_attention", "full_attention"]
            },
            "vision_config": { "model_type": "pixtral" }
        });
        assert_eq!(
            InferenceArchitecture::from_config(&config).unwrap(),
            InferenceArchitecture::Pixtral
        );
    }

    #[test]
    fn classifies_mistral3_pixtral_wrapper() {
        let config = json!({
            "model_type": "mistral3",
            "architectures": ["LightOnOCRForConditionalGeneration"],
            "text_config": { "model_type": "qwen3" },
            "vision_config": { "model_type": "pixtral" }
        });
        assert_eq!(
            InferenceArchitecture::from_config(&config).unwrap(),
            InferenceArchitecture::Pixtral
        );
    }

    #[test]
    fn classifies_qwen35_hybrid_decoder() {
        let config = json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention", "full_attention"]
            },
            "vision_config": { "model_type": "qwen3_5" }
        });
        assert_eq!(
            InferenceArchitecture::from_config(&config).unwrap(),
            InferenceArchitecture::Qwen
        );
    }

    #[test]
    fn rejects_unknown_wrapper() {
        let config = json!({
            "model_type": "unknown",
            "architectures": ["SomethingElse"],
            "text_config": { "model_type": "qwen3" },
            "vision_config": { "model_type": "pixtral" }
        });
        let error = InferenceArchitecture::from_config(&config).unwrap_err();
        match error {
            Error::UnsupportedConfiguration { reason } => {
                assert!(reason.contains("unsupported architecture"));
                assert!(reason.contains("unknown"));
            }
            other => panic!("expected unsupported configuration, got {other:?}"),
        }
    }

    #[test]
    fn rejects_qwen_without_layer_types() {
        let config = json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5" }
        });
        let error = InferenceArchitecture::from_config(&config).unwrap_err();
        match error {
            Error::UnsupportedConfiguration { reason } => {
                assert!(reason.contains("linear/full"));
            }
            other => panic!("expected unsupported configuration, got {other:?}"),
        }
    }

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
