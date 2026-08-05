//! Decoder key/value cache representation.
//!
//! The host [`KvCache`] is always available. With `--features cuda`, this module
//! also provides [`CudaKvState`] for device-resident past/present tensors used
//! by the decoder IoBinding path.

use super::DecoderConfig;

#[cfg(feature = "cuda")]
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
#[cfg(feature = "cuda")]
use ort::session::Session;
#[cfg(feature = "cuda")]
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};

#[cfg(feature = "cuda")]
use crate::{Error, Result};

/// Key/value tensors for one decoder layer.
///
/// Each tensor stores contiguous `float32` data with shape
/// `(batch_size, num_key_value_heads, past_sequence_length, head_dim)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerCache {
    key: Vec<f32>,
    value: Vec<f32>,
}

impl LayerCache {
    /// Creates one decoder layer cache from raw key and value tensor data.
    pub fn new(key: Vec<f32>, value: Vec<f32>) -> Self {
        Self { key, value }
    }

    /// Returns the cached key tensor data.
    pub fn key(&self) -> &[f32] {
        &self.key
    }

    /// Returns the cached value tensor data.
    pub fn value(&self) -> &[f32] {
        &self.value
    }

    pub(crate) fn key_mut(&mut self) -> &mut Vec<f32> {
        &mut self.key
    }

    pub(crate) fn value_mut(&mut self) -> &mut Vec<f32> {
        &mut self.value
    }
}

/// Opaque decoder key/value cache passed between decoder invocations.
///
/// `KvCache` contains one [`LayerCache`] for each decoder layer. The cache is
/// initialized empty for the first decoder pass. During autoregressive
/// generation the decoder overwrites each layer's buffers in place, reusing
/// allocation capacity across steps.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    batch_size: usize,
    past_sequence_length: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Creates an empty KV cache for a decoder configuration and batch size.
    pub fn empty(config: &DecoderConfig, batch_size: usize) -> crate::Result<Self> {
        if batch_size == 0 {
            return Err(crate::Error::InvalidKvCache {
                reason: "batch size must be greater than zero".to_owned(),
            });
        }

        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerCache::new(Vec::new(), Vec::new()))
            .collect();

        Self::new(
            layers,
            batch_size,
            0,
            config.num_key_value_heads,
            config.head_dim,
        )
    }

    /// Creates a typed KV cache from raw per-layer key/value tensors.
    pub fn new(
        layers: Vec<LayerCache>,
        batch_size: usize,
        past_sequence_length: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> crate::Result<Self> {
        if batch_size == 0 {
            return Err(crate::Error::InvalidKvCache {
                reason: "batch size must be greater than zero".to_owned(),
            });
        }
        if num_key_value_heads == 0 {
            return Err(crate::Error::InvalidKvCache {
                reason: "KV head count must be greater than zero".to_owned(),
            });
        }
        if head_dim == 0 {
            return Err(crate::Error::InvalidKvCache {
                reason: "KV head dimension must be greater than zero".to_owned(),
            });
        }

        let expected = values_per_tensor(
            batch_size,
            past_sequence_length,
            num_key_value_heads,
            head_dim,
        )?;
        for (index, layer) in layers.iter().enumerate() {
            if layer.key.len() != expected {
                return Err(crate::Error::InvalidKvCache {
                    reason: format!(
                        "layer {index} key length {} does not match shape {:?}",
                        layer.key.len(),
                        (
                            batch_size,
                            num_key_value_heads,
                            past_sequence_length,
                            head_dim
                        )
                    ),
                });
            }
            if layer.value.len() != expected {
                return Err(crate::Error::InvalidKvCache {
                    reason: format!(
                        "layer {index} value length {} does not match shape {:?}",
                        layer.value.len(),
                        (
                            batch_size,
                            num_key_value_heads,
                            past_sequence_length,
                            head_dim
                        )
                    ),
                });
            }
        }

        Ok(Self {
            layers,
            batch_size,
            past_sequence_length,
            num_key_value_heads,
            head_dim,
        })
    }

    /// Returns all layer caches.
    pub fn layers(&self) -> &[LayerCache] {
        &self.layers
    }

    /// Returns the number of decoder layers represented by the cache.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the cached sequence length before the next decoder pass.
    pub fn past_sequence_length(&self) -> usize {
        self.past_sequence_length
    }

    /// Returns the number of key/value heads.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Returns the per-head key/value dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the per-layer tensor shape.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (
            self.batch_size,
            self.num_key_value_heads,
            self.past_sequence_length,
            self.head_dim,
        )
    }

    /// Returns whether the cache contains no past sequence positions.
    pub fn is_empty(&self) -> bool {
        self.past_sequence_length == 0
    }

    pub(crate) fn layers_mut(&mut self) -> &mut [LayerCache] {
        &mut self.layers
    }

    pub(crate) fn set_past_sequence_length(&mut self, past_sequence_length: usize) {
        self.past_sequence_length = past_sequence_length;
    }
}

pub(crate) fn values_per_tensor(
    batch_size: usize,
    sequence_length: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> crate::Result<usize> {
    batch_size
        .checked_mul(num_key_value_heads)
        .and_then(|value| value.checked_mul(sequence_length))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| crate::Error::InvalidKvCache {
            reason: "KV cache tensor shape is too large".to_owned(),
        })
}

/// Device-resident KV past tensors for CUDA IoBinding decode.
///
/// Compiled only with `--features cuda`. Not a public API twin of [`KvCache`];
/// it exists so the decoder can keep past/present on GPU across steps.
#[cfg(feature = "cuda")]
pub(crate) struct CudaKvState {
    pub past_keys: Vec<DynValue>,
    pub past_values: Vec<DynValue>,
    pub past_sequence_length: usize,
    pub batch_size: usize,
}

#[cfg(feature = "cuda")]
impl CudaKvState {
    /// Allocates empty `(batch, kv_heads, 0, head_dim)` past tensors on CUDA.
    pub(crate) fn empty(
        session: &Session,
        config: &DecoderConfig,
        batch_size: usize,
        device_id: i32,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidKvCache {
                reason: "batch size must be greater than zero".to_owned(),
            });
        }

        let allocator =
            Allocator::new(session, cuda_memory_info(device_id)?).map_err(|source| {
                Error::OnnxRuntimeCompatibility {
                    reason: format!("failed to create CUDA allocator: {source}"),
                }
            })?;

        let shape = [
            batch_size as i64,
            config.num_key_value_heads as i64,
            0_i64,
            config.head_dim as i64,
        ];

        let mut past_keys = Vec::with_capacity(config.num_hidden_layers);
        let mut past_values = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let key = Tensor::<f32>::new(&allocator, shape).map_err(|source| {
                Error::OnnxRuntimeCompatibility {
                    reason: format!("failed to allocate empty CUDA past key: {source}"),
                }
            })?;
            let value = Tensor::<f32>::new(&allocator, shape).map_err(|source| {
                Error::OnnxRuntimeCompatibility {
                    reason: format!("failed to allocate empty CUDA past value: {source}"),
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

    pub(crate) fn is_empty(&self) -> bool {
        self.past_sequence_length == 0
    }

    /// Replaces past buffers with present outputs and advances sequence length.
    pub(crate) fn promote_present(
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

#[cfg(feature = "cuda")]
pub(crate) fn cuda_memory_info(device_id: i32) -> Result<MemoryInfo<'static>> {
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

#[cfg(feature = "cuda")]
pub(crate) fn cpu_output_memory_info() -> Result<MemoryInfo<'static>> {
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

#[cfg(feature = "cuda")]
pub(crate) fn validate_device_cache_output(
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
