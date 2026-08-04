//! Decoder key/value cache representation.

use super::DecoderConfig;

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
/// initialized empty for the first decoder pass and replaced with the updated
/// cache returned by the ONNX decoder after every pass.
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

/// Experimental host-side KV cache update strategy.
///
/// Selected at runtime via `LIGHTONOCR_KV_STRATEGY` for investigation builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum KvUpdateStrategy {
    /// Allocate a fresh `Vec` for every present tensor (production baseline).
    #[default]
    FullCopyReplace,
    /// Reuse previous layer buffers when capacity allows; still copy full present.
    ReusableBuffers,
    /// Reuse buffers; copy retained positions from the previous cache and only
    /// the new token slice(s) from each present tensor.
    DeltaExtractOnly,
}

impl KvUpdateStrategy {
    pub(crate) fn from_env() -> Self {
        match std::env::var("LIGHTONOCR_KV_STRATEGY")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "reusable" | "reusable_buffers" => Self::ReusableBuffers,
            "delta" | "delta_extract" | "delta_extract_only" => Self::DeltaExtractOnly,
            _ => Self::FullCopyReplace,
        }
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
