//! Qwen autoregressive decoder with hybrid recurrent/KV state.
//!
//! The exported Qwen3.5 OCR graphs are split:
//!
//! - `decoder_prefill.onnx` consumes `inputs_embeds`, 3-axis `position_ids`,
//!   and `attention_mask`, and returns `logits` plus per-layer state;
//! - `decoder_decode.onnx` consumes the same three tensors for one new token
//!   plus the previous layer states, and returns updated states.
//!
//! Decode **inputs** use semantic names (`conv_state.{i}` /
//! `recurrent_state.{i}` or `past_key.{i}` / `past_value.{i}`). Graph
//! **outputs** are uniformly `state.{i}.0` and `state.{i}.1`. After prefill,
//! decode steps advance all three position axes by `sequence + rope_delta`.

use std::path::Path;

use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::TensorRef;

use crate::model::{AttentionMask, EmbeddingModel, InputEmbeddings};
use crate::util::RuntimeOptions;
use crate::{Error, Result};

use super::config::GenerationConfig;
use super::generation::{self, FinishReason, GenerationOutput};
use super::logits::Logits;

const INPUT_EMBEDS: &str = "inputs_embeds";
const POSITION_IDS: &str = "position_ids";
const ATTENTION_MASK: &str = "attention_mask";
const LOGITS: &str = "logits";

/// Attention implementation used by one Qwen decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenLayerType {
    /// Gated linear attention with convolution and recurrent state.
    LinearAttention,
    /// Full self-attention with key/value cache tensors.
    FullAttention,
}

/// Prompt metadata required by Qwen's multimodal rotary position encoding.
#[derive(Debug)]
pub(crate) struct QwenPromptState {
    position_ids: Vec<i64>,
    rope_delta: i64,
}

impl QwenPromptState {
    pub(crate) fn new(position_ids: Vec<i64>, rope_delta: i64) -> Self {
        Self {
            position_ids,
            rope_delta,
        }
    }
}

#[derive(Debug)]
struct StateTensor {
    shape: Vec<i64>,
    data: Vec<f32>,
}

/// ONNX Runtime wrapper for the exported Qwen prefill and decode graphs.
#[derive(Debug)]
pub struct QwenDecoder {
    prefill: Session,
    decode: Session,
    generation_config: GenerationConfig,
    layer_types: Vec<QwenLayerType>,
    hidden_size: usize,
    vocab_size: usize,
    rng: fastrand::Rng,
}

impl QwenDecoder {
    /// Loads the Qwen prefill and decode graphs and validates their contracts.
    pub fn from_model_path(
        prefill_path: impl AsRef<Path>,
        decode_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        generation_path: impl AsRef<Path>,
        runtime: &RuntimeOptions,
    ) -> Result<Self> {
        let prefill_path = prefill_path.as_ref().to_path_buf();
        let decode_path = decode_path.as_ref().to_path_buf();
        let config_path = config_path.as_ref();
        let config: serde_json::Value = crate::util::json::load_json_file(config_path)?;
        if config.get("model_type").and_then(serde_json::Value::as_str) != Some("qwen3_5") {
            return Err(Error::UnsupportedConfiguration {
                reason: format!("expected Qwen config at {}", config_path.display()),
            });
        }
        let text = config
            .get("text_config")
            .ok_or_else(|| Error::UnsupportedConfiguration {
                reason: "Qwen checkpoint is missing text_config".to_owned(),
            })?;
        let layer_types = parse_layer_types(text)?;
        let hidden_size = json_usize(text, "hidden_size")?;
        let vocab_size = json_usize(text, "vocab_size")?;
        let prefill = crate::util::onnxruntime::session_builder(runtime)?
            .commit_from_file(&prefill_path)
            .map_err(|source| Error::DecoderModelLoad {
                path: prefill_path.clone(),
                source,
            })?;
        let decode = crate::util::onnxruntime::session_builder(runtime)?
            .commit_from_file(&decode_path)
            .map_err(|source| Error::DecoderModelLoad {
                path: decode_path.clone(),
                source,
            })?;
        validate_graphs(&prefill, &decode, &layer_types)?;
        Ok(Self {
            prefill,
            decode,
            generation_config: GenerationConfig::from_qwen_file(generation_path)?,
            layer_types,
            hidden_size,
            vocab_size,
            rng: fastrand::Rng::new(),
        })
    }

    /// Returns the generation configuration loaded with this decoder.
    #[must_use]
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }

    /// Returns mutable access to the generation configuration.
    pub fn generation_config_mut(&mut self) -> &mut GenerationConfig {
        &mut self.generation_config
    }

    /// Runs prefill then cached decode, invoking `on_token` for each new id.
    pub(crate) fn generate(
        &mut self,
        input_embeddings: InputEmbeddings,
        mut attention_mask: AttentionMask,
        prompt_state: &QwenPromptState,
        embedding_model: &mut EmbeddingModel,
        mut on_token: impl FnMut(i64),
    ) -> Result<GenerationOutput> {
        if self.generation_config.max_new_tokens == 0 {
            return Ok(GenerationOutput::new(Vec::new(), FinishReason::Length));
        }
        let (batch, sequence, hidden) = input_embeddings.shape();
        if batch != 1
            || hidden != self.hidden_size
            || prompt_state.position_ids.len() != 3 * sequence
        {
            return Err(Error::Inference {
                reason: format!(
                    "Qwen input shape {:?} or position-ID length {} is invalid",
                    input_embeddings.shape(),
                    prompt_state.position_ids.len()
                ),
            });
        }
        attention_mask.reserve(self.generation_config.max_new_tokens);
        let prefill_outputs = self.run_prefill(
            &input_embeddings,
            &prompt_state.position_ids,
            &attention_mask,
        )?;
        let mut logits = prefill_outputs.0;
        let mut states = prefill_outputs.1;
        let mut generated = Vec::with_capacity(self.generation_config.max_new_tokens);
        let mut finish_reason = FinishReason::Length;
        let mut step_embeddings =
            InputEmbeddings::new(vec![0.0; self.hidden_size], 1, 1, self.hidden_size)?;
        for step in 0..self.generation_config.max_new_tokens {
            let token_id = generation::next_token(&self.generation_config, &logits, &mut self.rng)?;
            generated.push(token_id);
            on_token(token_id);
            if generation::is_eos(&self.generation_config, token_id) {
                finish_reason = FinishReason::EndOfSequence;
                break;
            }
            if step + 1 == self.generation_config.max_new_tokens {
                break;
            }
            embedding_model.embed_into(&[token_id], &mut step_embeddings)?;
            attention_mask.push_visible();
            // Prefill used 3-axis prompt positions. Decode continues that
            // timeline at `sequence + rope_delta`, then one step per token.
            let next_position = sequence as i64 + prompt_state.rope_delta + step as i64;
            let position = [next_position; 3];
            let (next_logits, next_states) =
                self.run_decode(&step_embeddings, &position, &attention_mask, &states)?;
            logits = next_logits;
            states = next_states;
        }
        Ok(GenerationOutput::new(generated, finish_reason))
    }

    fn run_prefill(
        &mut self,
        embeddings: &InputEmbeddings,
        position_ids: &[i64],
        attention_mask: &AttentionMask,
    ) -> Result<(Logits, Vec<StateTensor>)> {
        let (_, sequence, hidden) = embeddings.shape();
        let inputs = vec![
            tensor_input_f32(
                vec![1, sequence as i64, hidden as i64],
                embeddings.as_slice(),
            )?,
            tensor_input_i64(vec![3, 1, sequence as i64], position_ids)?,
            tensor_input_i64(vec![1, sequence as i64], attention_mask.as_slice())?,
        ];
        let mut outputs = self
            .prefill
            .run(inputs.as_slice())
            .map_err(|source| Error::DecoderInference { source })?;
        extract_outputs(&mut outputs, self.vocab_size, &self.layer_types)
    }

    fn run_decode(
        &mut self,
        embeddings: &InputEmbeddings,
        position: &[i64; 3],
        attention_mask: &AttentionMask,
        states: &[StateTensor],
    ) -> Result<(Logits, Vec<StateTensor>)> {
        let mut inputs = vec![
            tensor_input_f32(vec![1, 1, self.hidden_size as i64], embeddings.as_slice())?,
            tensor_input_i64(vec![3, 1, 1], position)?,
            tensor_input_i64(
                vec![1, attention_mask.len() as i64],
                attention_mask.as_slice(),
            )?,
        ];
        for state in states {
            inputs.push(tensor_input_f32(
                state.shape.clone(),
                state.data.as_slice(),
            )?);
        }
        let mut outputs = self
            .decode
            .run(inputs.as_slice())
            .map_err(|source| Error::DecoderInference { source })?;
        extract_outputs(&mut outputs, self.vocab_size, &self.layer_types)
    }
}

fn tensor_input_f32<'a>(shape: Vec<i64>, data: &'a [f32]) -> Result<SessionInputValue<'a>> {
    let tensor = TensorRef::from_array_view((shape, data))
        .map_err(|source| Error::DecoderTensorCreation { source })?;
    Ok(SessionInputValue::from(tensor))
}

fn tensor_input_i64<'a>(shape: Vec<i64>, data: &'a [i64]) -> Result<SessionInputValue<'a>> {
    let tensor = TensorRef::from_array_view((shape, data))
        .map_err(|source| Error::DecoderTensorCreation { source })?;
    Ok(SessionInputValue::from(tensor))
}

fn parse_layer_types(text: &serde_json::Value) -> Result<Vec<QwenLayerType>> {
    let layer_types = text
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::UnsupportedConfiguration {
            reason: "Qwen text_config is missing layer_types".to_owned(),
        })?;
    layer_types
        .iter()
        .map(|kind| match kind.as_str() {
            Some("linear_attention") => Ok(QwenLayerType::LinearAttention),
            Some("full_attention") => Ok(QwenLayerType::FullAttention),
            other => Err(Error::UnsupportedConfiguration {
                reason: format!("unsupported Qwen layer type: {other:?}"),
            }),
        })
        .collect()
}

fn extract_outputs(
    outputs: &mut SessionOutputs<'_>,
    vocab_size: usize,
    layer_types: &[QwenLayerType],
) -> Result<(Logits, Vec<StateTensor>)> {
    let logits_value = outputs
        .remove(LOGITS)
        .ok_or_else(|| Error::InvalidDecoderOutput {
            reason: "Qwen graph did not return logits".to_owned(),
        })?;
    let (shape, values) =
        logits_value
            .try_extract_tensor::<f32>()
            .map_err(|source| Error::InvalidDecoderOutput {
                reason: format!("could not read Qwen logits: {source}"),
            })?;
    let shape = shape.as_ref();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != vocab_size as i64 {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("invalid Qwen logits shape {shape:?}"),
        });
    }
    let logits = Logits::new(values.to_vec(), 1, shape[1] as usize, vocab_size)?;
    let mut states = Vec::with_capacity(layer_types.len() * 2);
    for layer in 0..layer_types.len() {
        for tensor_index in 0..2 {
            let output_name = format!("state.{layer}.{tensor_index}");
            let value = outputs.remove(output_name.as_str()).ok_or_else(|| {
                Error::InvalidDecoderOutput {
                    reason: format!("Qwen graph did not return `{output_name}`"),
                }
            })?;
            let (shape, values) = value.try_extract_tensor::<f32>().map_err(|source| {
                Error::InvalidDecoderOutput {
                    reason: format!("could not read `{output_name}`: {source}"),
                }
            })?;
            states.push(StateTensor {
                shape: shape.as_ref().to_vec(),
                data: values.to_vec(),
            });
        }
    }
    Ok((logits, states))
}

fn validate_graphs(
    prefill: &Session,
    decode: &Session,
    layer_types: &[QwenLayerType],
) -> Result<()> {
    let layers = layer_types.len();
    let prefill_inputs = prefill.inputs();
    let decode_inputs = decode.inputs();
    if prefill_inputs.len() != 3
        || prefill_inputs[0].name() != INPUT_EMBEDS
        || prefill_inputs[1].name() != POSITION_IDS
        || prefill_inputs[2].name() != ATTENTION_MASK
    {
        return Err(Error::InvalidDecoderModel {
            reason: "Qwen prefill input contract mismatch".to_owned(),
        });
    }
    if decode_inputs.len() != 3 + 2 * layers
        || decode_inputs[0].name() != INPUT_EMBEDS
        || decode_inputs[1].name() != POSITION_IDS
        || decode_inputs[2].name() != ATTENTION_MASK
    {
        return Err(Error::InvalidDecoderModel {
            reason: "Qwen decode input count/order mismatch".to_owned(),
        });
    }
    for (layer, kind) in layer_types.iter().enumerate() {
        let expected = match kind {
            QwenLayerType::LinearAttention => [
                format!("conv_state.{layer}"),
                format!("recurrent_state.{layer}"),
            ],
            QwenLayerType::FullAttention => {
                [format!("past_key.{layer}"), format!("past_value.{layer}")]
            }
        };
        let start = 3 + layer * 2;
        if decode_inputs[start].name() != expected[0]
            || decode_inputs[start + 1].name() != expected[1]
        {
            return Err(Error::InvalidDecoderModel {
                reason: format!("Qwen decode state order mismatch at layer {layer}"),
            });
        }
    }
    if prefill.outputs().len() != 1 + 2 * layers || decode.outputs().len() != 1 + 2 * layers {
        return Err(Error::InvalidDecoderModel {
            reason: "Qwen decoder output count mismatch".to_owned(),
        });
    }
    Ok(())
}

fn json_usize(object: &serde_json::Value, key: &str) -> Result<usize> {
    object
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::UnsupportedConfiguration {
            reason: format!("Qwen config field `{key}` is missing or invalid"),
        })
}
