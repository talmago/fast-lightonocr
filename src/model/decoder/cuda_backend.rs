//! CUDA-resident KV cache and IoBinding decode step.
//!
//! Compiled only with `--features cuda`. Host [`super::KVCache`] stays
//! device-unaware; this module is the CUDA implementation of
//! [`super::KVCacheBackend`]. See [`docs/KV.md`](../../../docs/KV.md).

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};

use crate::model::InputEmbeddings;
use crate::{Error, Result};

use super::attention::AttentionMask;
use super::config::DecoderConfig;
use super::decoder::{
    DECODER_ATTENTION_MASK_NAME, DECODER_INPUT_EMBEDS_NAME, DECODER_LOGITS_NAME,
    DECODER_USE_CACHE_BRANCH_NAME, LogitsSelection, extract_logits, past_key_name, past_value_name,
    present_key_name, present_value_name,
};
use super::kv_cache::KVCacheBackend;
use super::logits::Logits;

/// CUDA-resident KV past tensors for IoBinding decode.
///
/// Prefill starts with host-side empty (`seq=0`) past tensors; after the first
/// step, [`Self::promote_present`] keeps present outputs on device.
pub(crate) struct CudaKVCache {
    past_keys: Vec<DynValue>,
    past_values: Vec<DynValue>,
    past_sequence_length: usize,
    batch_size: usize,
}

impl CudaKVCache {
    /// Creates empty `(batch, kv_heads, 0, head_dim)` past tensors on the **host**.
    ///
    /// Zero-length past tensors are allocated on CPU on purpose: CUDA allocations
    /// with a zero sequence dimension are unreliable with ORT IoBinding and have
    /// been observed to segfault. After the first decode step, [`Self::promote_present`]
    /// replaces these with device-resident present tensors.
    pub(crate) fn empty(config: &DecoderConfig, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidKVCache {
                reason: "batch size must be greater than zero".to_owned(),
            });
        }

        let shape = [
            batch_size as i64,
            config.num_key_value_heads as i64,
            0_i64,
            config.head_dim as i64,
        ];

        let mut past_keys = Vec::with_capacity(config.num_hidden_layers);
        let mut past_values = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let key = Tensor::<f32>::from_array((shape, Vec::<f32>::new())).map_err(|source| {
                Error::OnnxRuntimeCompatibility {
                    reason: format!("failed to create empty host past key: {source}"),
                }
            })?;
            let value =
                Tensor::<f32>::from_array((shape, Vec::<f32>::new())).map_err(|source| {
                    Error::OnnxRuntimeCompatibility {
                        reason: format!("failed to create empty host past value: {source}"),
                    }
                })?;
            past_keys.push(key.into_dyn());
            past_values.push(value.into_dyn());
        }

        Ok(Self {
            past_keys,
            past_values,
            past_sequence_length: 0,
            batch_size,
        })
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn past_sequence_length(&self) -> usize {
        self.past_sequence_length
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.past_sequence_length == 0
    }

    fn past_keys(&self) -> &[DynValue] {
        &self.past_keys
    }

    fn past_values(&self) -> &[DynValue] {
        &self.past_values
    }

    /// Replaces past buffers with present outputs and advances sequence length.
    fn promote_present(
        &mut self,
        past_keys: Vec<DynValue>,
        past_values: Vec<DynValue>,
        total_sequence_length: usize,
    ) {
        self.past_keys = past_keys;
        self.past_values = past_values;
        self.past_sequence_length = total_sequence_length;
    }
}

impl KVCacheBackend for CudaKVCache {
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn past_sequence_length(&self) -> usize {
        self.past_sequence_length
    }

    fn is_empty(&self) -> bool {
        self.past_sequence_length == 0
    }
}

/// ORT memory infos and reusable host staging buffers for CUDA IoBinding decode.
pub(crate) struct CudaIoContext {
    cuda_mem: MemoryInfo<'static>,
    cpu_mem: MemoryInfo<'static>,
    embeds_staging: Vec<f32>,
    mask_staging: Vec<i64>,
}

impl CudaIoContext {
    pub(crate) fn new(device_id: i32) -> Result<Self> {
        Ok(Self {
            cuda_mem: cuda_memory_info(device_id)?,
            cpu_mem: cpu_output_memory_info()?,
            embeds_staging: Vec::new(),
            mask_staging: Vec::new(),
        })
    }

    /// Copies `data` into a reusable staging buffer, then builds an owned ORT tensor.
    ///
    /// Staging capacity is retained across steps so host reallocations stay rare.
    fn embeds_tensor(&mut self, shape: [i64; 3], data: &[f32]) -> Result<Tensor<f32>> {
        self.embeds_staging.clear();
        self.embeds_staging.extend_from_slice(data);
        Tensor::<f32>::from_array((shape, self.embeds_staging.clone()))
            .map_err(|source| Error::DecoderTensorCreation { source })
    }

    /// Copies mask data into a reusable staging buffer, then builds an owned ORT tensor.
    fn mask_tensor(&mut self, shape: [i64; 2], data: &[i64]) -> Result<Tensor<i64>> {
        self.mask_staging.clear();
        self.mask_staging.extend_from_slice(data);
        Tensor::<i64>::from_array((shape, self.mask_staging.clone()))
            .map_err(|source| Error::DecoderTensorCreation { source })
    }
}

fn cuda_memory_info(device_id: i32) -> Result<MemoryInfo<'static>> {
    MemoryInfo::new(
        AllocationDevice::CUDA,
        device_id,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|source| Error::OnnxRuntimeCompatibility {
        reason: format!("failed to create CUDA MemoryInfo: {source}"),
    })
}

fn cpu_output_memory_info() -> Result<MemoryInfo<'static>> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )
    .map_err(|source| Error::OnnxRuntimeCompatibility {
        reason: format!("failed to create CPU output MemoryInfo: {source}"),
    })
}

fn validate_device_cache_output(
    value: &DynValue,
    name: &str,
    batch_size: usize,
    total_sequence_length: usize,
    config: &DecoderConfig,
) -> Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{name}` is not a tensor"),
        });
    };
    if *ty != TensorElementType::Float32 {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{name}` has element type {ty:?}, expected Float32"),
        });
    }
    if shape.len() != 4 {
        return Err(Error::InvalidDecoderOutput {
            reason: format!("`{name}` has rank {}, expected 4", shape.len()),
        });
    }
    let expected = [
        batch_size as i64,
        config.num_key_value_heads as i64,
        total_sequence_length as i64,
        config.head_dim as i64,
    ];
    for (axis, expected_dim) in expected.into_iter().enumerate() {
        if shape[axis] != expected_dim {
            return Err(Error::InvalidDecoderOutput {
                reason: format!(
                    "`{name}` dimension {axis} is {}, expected {expected_dim}",
                    shape[axis]
                ),
            });
        }
    }
    Ok(())
}

/// Inputs for one CUDA IoBinding decoder step.
pub(crate) struct CudaDecodeStep<'a> {
    pub session: &'a mut Session,
    pub config: &'a DecoderConfig,
    pub has_cache_branch: bool,
    pub input_embeddings: &'a InputEmbeddings,
    pub attention_mask: &'a AttentionMask,
    pub kv_cache: &'a mut CudaKVCache,
    pub cuda_io: &'a mut CudaIoContext,
    pub logits_selection: LogitsSelection,
    pub logits_scratch: &'a mut Vec<f32>,
}

/// IoBinding decode: past/present stay on device after prefill; logits on host.
pub(crate) fn decode_step(step: CudaDecodeStep<'_>) -> Result<Logits> {
    let CudaDecodeStep {
        session,
        config,
        has_cache_branch,
        input_embeddings,
        attention_mask,
        kv_cache,
        cuda_io,
        logits_selection,
        logits_scratch,
    } = step;
    let (batch_size, sequence_length, hidden_size) = input_embeddings.shape();
    if batch_size != kv_cache.batch_size() {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "input embeddings batch size is {batch_size}, expected {}",
                kv_cache.batch_size()
            ),
        });
    }
    if hidden_size != config.hidden_size {
        return Err(Error::InvalidDecoderInput {
            reason: format!(
                "input embeddings hidden size is {hidden_size}, expected {}",
                config.hidden_size
            ),
        });
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

    let use_cache_branch = !kv_cache.is_empty();

    let embeds = cuda_io.embeds_tensor(
        [
            batch_size as i64,
            sequence_length as i64,
            hidden_size as i64,
        ],
        input_embeddings.as_slice(),
    )?;
    let mask = cuda_io.mask_tensor(
        [batch_size as i64, total_sequence_length as i64],
        attention_mask.as_slice(),
    )?;

    let mut binding = session
        .create_binding()
        .map_err(|source| Error::DecoderInference { source })?;

    binding
        .bind_input(DECODER_INPUT_EMBEDS_NAME, &embeds)
        .map_err(|source| Error::DecoderInference { source })?;
    binding
        .bind_input(DECODER_ATTENTION_MASK_NAME, &mask)
        .map_err(|source| Error::DecoderInference { source })?;

    let cache_branch_tensor = if has_cache_branch {
        let tensor = Tensor::from_array(((), vec![use_cache_branch]))
            .map_err(|source| Error::DecoderTensorCreation { source })?;
        binding
            .bind_input(DECODER_USE_CACHE_BRANCH_NAME, &tensor)
            .map_err(|source| Error::DecoderInference { source })?;
        Some(tensor)
    } else {
        None
    };
    let _cache_branch_tensor = cache_branch_tensor;

    for layer_index in 0..config.num_hidden_layers {
        binding
            .bind_input(
                past_key_name(layer_index),
                &kv_cache.past_keys()[layer_index],
            )
            .map_err(|source| Error::DecoderInference { source })?;
        binding
            .bind_input(
                past_value_name(layer_index),
                &kv_cache.past_values()[layer_index],
            )
            .map_err(|source| Error::DecoderInference { source })?;
    }

    binding
        .bind_output_to_device(DECODER_LOGITS_NAME, &cuda_io.cpu_mem)
        .map_err(|source| Error::DecoderInference { source })?;
    for layer_index in 0..config.num_hidden_layers {
        binding
            .bind_output_to_device(present_key_name(layer_index), &cuda_io.cuda_mem)
            .map_err(|source| Error::DecoderInference { source })?;
        binding
            .bind_output_to_device(present_value_name(layer_index), &cuda_io.cuda_mem)
            .map_err(|source| Error::DecoderInference { source })?;
    }

    let mut outputs = session
        .run_binding(&binding)
        .map_err(|source| Error::DecoderInference { source })?;

    let logits = extract_logits(
        &mut outputs,
        batch_size,
        sequence_length,
        config.vocab_size,
        logits_selection,
        logits_scratch,
    )?;

    let mut next_keys = Vec::with_capacity(config.num_hidden_layers);
    let mut next_values = Vec::with_capacity(config.num_hidden_layers);
    for layer_index in 0..config.num_hidden_layers {
        let key_name = present_key_name(layer_index);
        let key = outputs
            .remove(key_name.as_str())
            .ok_or_else(|| Error::InvalidDecoderOutput {
                reason: format!("missing `{key_name}` output"),
            })?;
        validate_device_cache_output(&key, &key_name, batch_size, total_sequence_length, config)?;

        let value_name = present_value_name(layer_index);
        let value =
            outputs
                .remove(value_name.as_str())
                .ok_or_else(|| Error::InvalidDecoderOutput {
                    reason: format!("missing `{value_name}` output"),
                })?;
        validate_device_cache_output(
            &value,
            &value_name,
            batch_size,
            total_sequence_length,
            config,
        )?;

        next_keys.push(key);
        next_values.push(value);
    }

    kv_cache.promote_present(next_keys, next_values, total_sequence_length);
    Ok(logits)
}
