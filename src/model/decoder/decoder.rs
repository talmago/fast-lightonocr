//! ONNX Runtime wrapper for the LightOnOCR decoder.

use std::path::{Path, PathBuf};

use ort::session::{Session, SessionInputValue};
use ort::value::{Outlet, TensorElementType, TensorRef, ValueType};

use crate::model::InputEmbeddings;
use crate::model::embedding_model::EmbeddingModel;
use crate::{Error, Result};

use super::attention::AttentionMask;
use super::config::{DecoderConfig, GenerationConfig};
use super::generation::{self, FinishReason, GenerationOutput};
use super::kv_cache::{KvCache, LayerCache, values_per_tensor};
use super::logits::Logits;
use super::output::DecoderOutput;

const DECODER_INPUT_EMBEDS_NAME: &str = "inputs_embeds";
const DECODER_ATTENTION_MASK_NAME: &str = "attention_mask";
const DECODER_USE_CACHE_BRANCH_NAME: &str = "use_cache_branch";
const DECODER_LOGITS_NAME: &str = "logits";

const CONFIG_FILE: &str = "config.json";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";

/// ONNX Runtime wrapper for the exported LightOnOCR decoder.
///
/// `Decoder` owns the decoder model together with the runtime state required
/// for autoregressive generation. It executes the decoder ONNX graph,
/// maintains the KV-cache across decoding steps, and generates output tokens
/// according to the configured generation strategy.
#[derive(Debug)]
pub struct Decoder {
    session: Session,
    config: DecoderConfig,
    generation_config: GenerationConfig,
    model_path: PathBuf,
    has_cache_branch: bool,
    rng: fastrand::Rng,
}

impl Decoder {
    /// Loads a decoder from an explicit ONNX model path.
    ///
    /// The decoder configuration (`config.json`) and generation configuration
    /// (`generation_config.json`) are loaded automatically from either the
    /// ONNX model directory or its parent model directory.
    pub fn from_model_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.is_file() {
            return Err(Error::MissingDecoderModel { path: model_path });
        }

        let config_dir = config_dir_for_model(&model_path)?;

        let config = DecoderConfig::from_file(config_dir.join(CONFIG_FILE))?;

        let generation_config =
            GenerationConfig::from_file(config_dir.join(GENERATION_CONFIG_FILE))?;

        crate::util::onnxruntime::ensure_compatible()?;

        let session = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(&model_path))
            .map_err(|source| Error::DecoderModelLoad {
                path: model_path.clone(),
                source,
            })?;

        let has_cache_branch = validate_session_contract(&session, &config)?;

        Ok(Self {
            session,
            config,
            generation_config,
            model_path,
            has_cache_branch,
            rng: fastrand::Rng::new(),
        })
    }

    /// Returns the loaded decoder configuration.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Returns the loaded generation configuration.
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }

    /// Returns mutable access to the generation configuration.
    pub fn generation_config_mut(&mut self) -> &mut GenerationConfig {
        &mut self.generation_config
    }

    /// Returns the ONNX model path backing this decoder.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Creates an empty KV cache for this decoder and batch size.
    pub fn empty_kv_cache(&self, batch_size: usize) -> Result<KvCache> {
        KvCache::empty(&self.config, batch_size)
    }

    /// Returns the image placeholder token id used by the decoder config.
    #[must_use]
    pub fn image_token_id(&self) -> i64 {
        self.config.image_token_index
    }

    /// Generates tokens autoregressively.
    pub fn generate(
        &mut self,
        decoder_input: InputEmbeddings,
        attention_mask: AttentionMask,
        embedding_model: &mut EmbeddingModel,
    ) -> Result<GenerationOutput> {
        self.generate_streaming(decoder_input, attention_mask, embedding_model, |_| {})
    }

    /// Generates tokens while streaming every generated token.
    pub fn generate_streaming(
        &mut self,
        mut decoder_input: InputEmbeddings,
        mut attention_mask: AttentionMask,
        embedding_model: &mut EmbeddingModel,
        mut on_token: impl FnMut(i64),
    ) -> Result<GenerationOutput> {
        if self.generation_config.max_new_tokens == 0 {
            return Ok(GenerationOutput::new(Vec::new(), FinishReason::Length));
        }

        let mut generated = Vec::with_capacity(self.generation_config.max_new_tokens);

        let mut kv_cache = self.empty_kv_cache(decoder_input.batch_size())?;
        attention_mask.reserve(self.generation_config.max_new_tokens);

        let mut finish_reason = FinishReason::Length;

        for step in 0..self.generation_config.max_new_tokens {
            let output = self.decode(&decoder_input, &attention_mask, &kv_cache)?;

            kv_cache = output.kv_cache;

            let next_token =
                generation::next_token(&self.generation_config, &output.logits, &mut self.rng)?;

            generated.push(next_token);
            on_token(next_token);

            if generation::is_eos(&self.generation_config, next_token) {
                finish_reason = FinishReason::EndOfSequence;
                break;
            }

            if step + 1 == self.generation_config.max_new_tokens {
                break;
            }

            decoder_input = embedding_model.embed(&[next_token])?;
            if step == 0 {
                attention_mask.fill_visible();
            }
            attention_mask.push_visible();
        }

        Ok(GenerationOutput::new(generated, finish_reason))
    }

    /// Executes one decoder pass.
    ///
    /// `input_embeddings` has shape `(batch_size, sequence_length,
    /// hidden_size)`. `attention_mask` is interpreted as shape
    /// `(batch_size, past_sequence_length + sequence_length)`. The returned
    /// [`DecoderOutput`] contains logits for the provided sequence positions
    /// and a cache covering the total sequence length.
    pub fn decode(
        &mut self,
        input_embeddings: &InputEmbeddings,
        attention_mask: &AttentionMask,
        kv_cache: &KvCache,
    ) -> Result<DecoderOutput> {
        let total_sequence_length =
            validate_decoder_inputs(input_embeddings, attention_mask, kv_cache, &self.config)?;

        let (batch_size, sequence_length, hidden_size) = input_embeddings.shape();
        let input_embeddings_shape = [
            batch_size as i64,
            sequence_length as i64,
            hidden_size as i64,
        ];
        let input_embeddings_tensor =
            TensorRef::from_array_view((input_embeddings_shape, input_embeddings.as_slice()))
                .map_err(|source| Error::DecoderTensorCreation { source })?;
        let attention_mask_shape = [batch_size as i64, total_sequence_length as i64];
        let attention_mask_tensor =
            TensorRef::from_array_view((attention_mask_shape, attention_mask.as_slice()))
                .map_err(|source| Error::DecoderTensorCreation { source })?;
        let use_cache_branch_value = [!kv_cache.is_empty()];

        let mut inputs =
            Vec::with_capacity(decoder_input_count(&self.config, self.has_cache_branch));
        inputs.push(SessionInputValue::from(input_embeddings_tensor));
        inputs.push(SessionInputValue::from(attention_mask_tensor));

        if self.has_cache_branch {
            let use_cache_branch_tensor =
                TensorRef::from_array_view(((), use_cache_branch_value.as_slice()))
                    .map_err(|source| Error::DecoderTensorCreation { source })?;
            inputs.push(SessionInputValue::from(use_cache_branch_tensor));
        }

        let cache_shape = [
            kv_cache.batch_size() as i64,
            kv_cache.num_key_value_heads() as i64,
            kv_cache.past_sequence_length() as i64,
            kv_cache.head_dim() as i64,
        ];
        for layer in kv_cache.layers() {
            let key_tensor = TensorRef::from_array_view((cache_shape, layer.key()))
                .map_err(|source| Error::DecoderTensorCreation { source })?;
            inputs.push(SessionInputValue::from(key_tensor));

            let value_tensor = TensorRef::from_array_view((cache_shape, layer.value()))
                .map_err(|source| Error::DecoderTensorCreation { source })?;
            inputs.push(SessionInputValue::from(value_tensor));
        }

        let mut outputs = self
            .session
            .run(inputs.as_slice())
            .map_err(|source| Error::DecoderInference { source })?;

        let logits_output =
            outputs
                .remove(DECODER_LOGITS_NAME)
                .ok_or_else(|| Error::InvalidDecoderOutput {
                    reason: format!("missing `{DECODER_LOGITS_NAME}` output"),
                })?;
        let (logits_shape, logits_data) =
            logits_output
                .try_extract_tensor::<f32>()
                .map_err(|source| Error::InvalidDecoderOutput {
                    reason: format!(
                        "failed to extract `{DECODER_LOGITS_NAME}` as float32: {source}"
                    ),
                })?;
        let logits_shape = logits_shape.as_ref();
        validate_logits_shape(
            logits_shape,
            batch_size,
            sequence_length,
            self.config.vocab_size,
        )?;
        let logits = Logits::new(
            logits_data.to_vec(),
            usize::try_from(logits_shape[0]).expect("validated non-negative batch size"),
            usize::try_from(logits_shape[1]).expect("validated non-negative sequence length"),
            usize::try_from(logits_shape[2]).expect("validated non-negative vocabulary size"),
        )?;

        let updated_cache = extract_updated_cache(
            &mut outputs,
            batch_size,
            total_sequence_length,
            &self.config,
        )?;

        Ok(DecoderOutput::new(logits, updated_cache))
    }
}

fn config_dir_for_model(model_path: &Path) -> Result<PathBuf> {
    let model_dir = model_path.parent().ok_or_else(|| Error::Inference {
        reason: format!(
            "decoder model path '{}' has no parent directory",
            model_path.display(),
        ),
    })?;

    let colocated_config = model_dir.join(CONFIG_FILE);
    let colocated_generation_config = model_dir.join(GENERATION_CONFIG_FILE);
    if colocated_config.is_file() && colocated_generation_config.is_file() {
        return Ok(model_dir.to_path_buf());
    }

    model_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::Inference {
            reason: format!(
                "decoder model path '{}' has no model root containing configuration files",
                model_path.display(),
            ),
        })
}

fn decoder_input_count(config: &DecoderConfig, has_cache_branch: bool) -> usize {
    2 + usize::from(has_cache_branch) + config.num_hidden_layers * 2
}

fn validate_session_contract(session: &Session, config: &DecoderConfig) -> Result<bool> {
    let inputs = session.inputs();
    let has_cache_branch = inputs
        .iter()
        .any(|input| input.name() == DECODER_USE_CACHE_BRANCH_NAME);
    let expected_input_count = decoder_input_count(config, has_cache_branch);
    if inputs.len() != expected_input_count {
        return Err(Error::InvalidDecoderModel {
            reason: format!(
                "expected {expected_input_count} inputs, found {}",
                inputs.len()
            ),
        });
    }

    validate_decoder_input_order(inputs, config, has_cache_branch)?;
    validate_named_tensor(
        inputs,
        DECODER_INPUT_EMBEDS_NAME,
        TensorElementType::Float32,
        &[3],
        &[(2, config.hidden_size)],
    )?;
    validate_named_tensor(
        inputs,
        DECODER_ATTENTION_MASK_NAME,
        TensorElementType::Int64,
        &[2],
        &[],
    )?;
    if has_cache_branch {
        validate_named_tensor(
            inputs,
            DECODER_USE_CACHE_BRANCH_NAME,
            TensorElementType::Bool,
            &[0, 1],
            &[],
        )?;
    }
    for layer_index in 0..config.num_hidden_layers {
        validate_named_tensor(
            inputs,
            &past_key_name(layer_index),
            TensorElementType::Float32,
            &[4],
            &[(1, config.num_key_value_heads), (3, config.head_dim)],
        )?;
        validate_named_tensor(
            inputs,
            &past_value_name(layer_index),
            TensorElementType::Float32,
            &[4],
            &[(1, config.num_key_value_heads), (3, config.head_dim)],
        )?;
    }

    let outputs = session.outputs();
    let expected_output_count = 1 + config.num_hidden_layers * 2;
    if outputs.len() != expected_output_count {
        return Err(Error::InvalidDecoderModel {
            reason: format!(
                "expected {expected_output_count} outputs, found {}",
                outputs.len()
            ),
        });
    }
    validate_named_tensor(
        outputs,
        DECODER_LOGITS_NAME,
        TensorElementType::Float32,
        &[3],
        &[(2, config.vocab_size)],
    )?;
    for layer_index in 0..config.num_hidden_layers {
        validate_named_tensor(
            outputs,
            &present_key_name(layer_index),
            TensorElementType::Float32,
            &[4],
            &[(1, config.num_key_value_heads), (3, config.head_dim)],
        )?;
        validate_named_tensor(
            outputs,
            &present_value_name(layer_index),
            TensorElementType::Float32,
            &[4],
            &[(1, config.num_key_value_heads), (3, config.head_dim)],
        )?;
    }

    Ok(has_cache_branch)
}

fn validate_decoder_input_order(
    inputs: &[Outlet],
    config: &DecoderConfig,
    has_cache_branch: bool,
) -> Result<()> {
    validate_input_at(inputs, 0, DECODER_INPUT_EMBEDS_NAME)?;
    validate_input_at(inputs, 1, DECODER_ATTENTION_MASK_NAME)?;

    let mut index = 2;
    if has_cache_branch {
        validate_input_at(inputs, index, DECODER_USE_CACHE_BRANCH_NAME)?;
        index += 1;
    }

    for layer_index in 0..config.num_hidden_layers {
        validate_input_at(inputs, index, &past_key_name(layer_index))?;
        index += 1;
        validate_input_at(inputs, index, &past_value_name(layer_index))?;
        index += 1;
    }

    Ok(())
}

fn validate_input_at(inputs: &[Outlet], index: usize, expected: &str) -> Result<()> {
    let actual = inputs
        .get(index)
        .ok_or_else(|| Error::InvalidDecoderModel {
            reason: format!("missing decoder input #{index}, expected `{expected}`"),
        })?
        .name();

    if actual != expected {
        return Err(Error::InvalidDecoderModel {
            reason: format!(
                "decoder input #{index} is `{actual}`, expected `{expected}`; positional execution depends on ONNX input order"
            ),
        });
    }

    Ok(())
}

fn validate_named_tensor(
    outlets: &[Outlet],
    name: &str,
    expected_type: TensorElementType,
    expected_ranks: &[usize],
    static_dims: &[(usize, usize)],
) -> Result<()> {
    let outlet = outlets
        .iter()
        .find(|outlet| outlet.name() == name)
        .ok_or_else(|| Error::InvalidDecoderModel {
            reason: format!("missing `{name}` tensor"),
        })?;
    validate_tensor_metadata(
        outlet.dtype(),
        name,
        expected_type,
        expected_ranks,
        static_dims,
    )
}

fn validate_tensor_metadata(
    value_type: &ValueType,
    name: &str,
    expected_type: TensorElementType,
    expected_ranks: &[usize],
    static_dims: &[(usize, usize)],
) -> Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        return Err(Error::InvalidDecoderModel {
            reason: format!("`{name}` is not a tensor"),
        });
    };
    if *ty != expected_type {
        return Err(Error::InvalidDecoderModel {
            reason: format!("`{name}` has type {ty}, expected {expected_type}"),
        });
    }
    if !expected_ranks.contains(&shape.len()) {
        return Err(Error::InvalidDecoderModel {
            reason: format!(
                "`{name}` has rank {}, expected one of {expected_ranks:?}",
                shape.len()
            ),
        });
    }
    for &(axis, expected) in static_dims {
        if axis >= shape.len() {
            return Err(Error::InvalidDecoderModel {
                reason: format!("`{name}` is missing static dimension axis {axis}"),
            });
        }
        let actual = shape[axis];
        if actual >= 0 && actual as usize != expected {
            return Err(Error::InvalidDecoderModel {
                reason: format!("`{name}` dimension {axis} is {actual}, expected {expected}"),
            });
        }
    }

    Ok(())
}

fn validate_decoder_inputs(
    input_embeddings: &InputEmbeddings,
    attention_mask: &AttentionMask,
    kv_cache: &KvCache,
    config: &DecoderConfig,
) -> Result<usize> {
    let (batch_size, sequence_length, hidden_size) = input_embeddings.shape();
    if batch_size == 0 {
        return Err(Error::InvalidDecoderInput {
            reason: "batch size must be greater than zero".to_owned(),
        });
    }
    if sequence_length == 0 {
        return Err(Error::InvalidDecoderInput {
            reason: "sequence length must be greater than zero".to_owned(),
        });
    }
    if hidden_size != config.hidden_size {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "hidden size is {hidden_size}, expected {}",
                config.hidden_size
            ),
        });
    }
    if kv_cache.layer_count() != config.num_hidden_layers {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "KV cache has {} layers, expected {}",
                kv_cache.layer_count(),
                config.num_hidden_layers
            ),
        });
    }
    if kv_cache.batch_size() != batch_size {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "KV cache batch size is {}, expected {batch_size}",
                kv_cache.batch_size()
            ),
        });
    }
    if kv_cache.num_key_value_heads() != config.num_key_value_heads {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "KV cache head count is {}, expected {}",
                kv_cache.num_key_value_heads(),
                config.num_key_value_heads
            ),
        });
    }
    if kv_cache.head_dim() != config.head_dim {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "KV cache head dimension is {}, expected {}",
                kv_cache.head_dim(),
                config.head_dim
            ),
        });
    }

    let expected_cache_values = values_per_tensor(
        kv_cache.batch_size(),
        kv_cache.past_sequence_length(),
        kv_cache.num_key_value_heads(),
        kv_cache.head_dim(),
    )
    .map_err(|error| Error::InvalidDecoderInput {
        reason: error.to_string(),
    })?;
    for (layer_index, layer) in kv_cache.layers().iter().enumerate() {
        if layer.key().len() != expected_cache_values {
            return Err(Error::InvalidDecoderInput {
                reason: format!(
                    "KV cache layer {layer_index} key length is {}, expected {expected_cache_values}",
                    layer.key().len()
                ),
            });
        }
        if layer.value().len() != expected_cache_values {
            return Err(Error::InvalidDecoderInput {
                reason: format!(
                    "KV cache layer {layer_index} value length is {}, expected {expected_cache_values}",
                    layer.value().len()
                ),
            });
        }
    }

    let total_sequence_length = kv_cache
        .past_sequence_length()
        .checked_add(sequence_length)
        .ok_or_else(|| Error::InvalidDecoderInput {
            reason: "total sequence length is too large".to_owned(),
        })?;
    let expected_attention_values =
        batch_size
            .checked_mul(total_sequence_length)
            .ok_or_else(|| Error::InvalidDecoderInput {
                reason: "attention mask shape is too large".to_owned(),
            })?;
    if attention_mask.len() != expected_attention_values {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "attention mask length is {}, expected {expected_attention_values}",
                attention_mask.len()
            ),
        });
    }

    Ok(total_sequence_length)
}

fn extract_updated_cache(
    outputs: &mut ort::session::SessionOutputs<'_>,
    batch_size: usize,
    total_sequence_length: usize,
    config: &DecoderConfig,
) -> Result<KvCache> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_index in 0..config.num_hidden_layers {
        let key_name = present_key_name(layer_index);
        let (key_shape, key_data) = extract_f32_output(outputs, &key_name)?;
        validate_cache_output_shape(
            &key_shape,
            &key_name,
            batch_size,
            total_sequence_length,
            config,
        )?;

        let value_name = present_value_name(layer_index);
        let (value_shape, value_data) = extract_f32_output(outputs, &value_name)?;
        validate_cache_output_shape(
            &value_shape,
            &value_name,
            batch_size,
            total_sequence_length,
            config,
        )?;

        layers.push(LayerCache::new(key_data, value_data));
    }

    KvCache::new(
        layers,
        batch_size,
        total_sequence_length,
        config.num_key_value_heads,
        config.head_dim,
    )
}

fn extract_f32_output(
    outputs: &mut ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let output = outputs
        .remove(name)
        .ok_or_else(|| Error::InvalidDecoderOutput {
            reason: format!("missing `{name}` output"),
        })?;
    let (shape, data) =
        output
            .try_extract_tensor::<f32>()
            .map_err(|source| Error::InvalidDecoderOutput {
                reason: format!("failed to extract `{name}` as float32: {source}"),
            })?;

    Ok((shape.as_ref().to_vec(), data.to_vec()))
}

fn validate_logits_shape(
    shape: &[i64],
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
) -> Result<()> {
    if shape.len() != 3 {
        return Err(Error::InvalidDecoderOutput {
            reason: format!(
                "`{DECODER_LOGITS_NAME}` has rank {}, expected 3",
                shape.len()
            ),
        });
    }
    if shape.iter().any(|&dim| dim < 0) {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{DECODER_LOGITS_NAME}` has negative shape {shape:?}"),
        });
    }
    if shape[0] as usize != batch_size {
        return Err(Error::InvalidDecoderOutput {
            reason: format!(
                "`{DECODER_LOGITS_NAME}` batch size is {}, expected {batch_size}",
                shape[0]
            ),
        });
    }
    if shape[1] as usize != sequence_length {
        return Err(Error::InvalidDecoderOutput {
            reason: format!(
                "`{DECODER_LOGITS_NAME}` sequence length is {}, expected {sequence_length}",
                shape[1]
            ),
        });
    }
    if shape[2] as usize != vocab_size {
        return Err(Error::InvalidDecoderOutput {
            reason: format!(
                "`{DECODER_LOGITS_NAME}` vocabulary size is {}, expected {vocab_size}",
                shape[2]
            ),
        });
    }

    Ok(())
}

fn validate_cache_output_shape(
    shape: &[i64],
    name: &str,
    batch_size: usize,
    total_sequence_length: usize,
    config: &DecoderConfig,
) -> Result<()> {
    if shape.len() != 4 {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{name}` has rank {}, expected 4", shape.len()),
        });
    }
    if shape.iter().any(|&dim| dim < 0) {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{name}` has negative shape {shape:?}"),
        });
    }
    let expected = [
        batch_size,
        config.num_key_value_heads,
        total_sequence_length,
        config.head_dim,
    ];
    for (axis, expected) in expected.into_iter().enumerate() {
        if shape[axis] as usize != expected {
            return Err(Error::InvalidDecoderOutput {
                reason: format!(
                    "`{name}` dimension {axis} is {}, expected {expected}",
                    shape[axis]
                ),
            });
        }
    }

    Ok(())
}

fn past_key_name(layer_index: usize) -> String {
    format!("past_key_values.{layer_index}.key")
}

fn past_value_name(layer_index: usize) -> String {
    format!("past_key_values.{layer_index}.value")
}

fn present_key_name(layer_index: usize) -> String {
    format!("present.{layer_index}.key")
}

fn present_value_name(layer_index: usize) -> String {
    format!("present.{layer_index}.value")
}
