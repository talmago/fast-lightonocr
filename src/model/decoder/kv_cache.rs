//! Decoder key/value cache representation.
//!
//! The host [`KVCache`] is always available and implements [`KVCacheBackend`].
//! With `--features cuda`, [`ActiveKVCache`] can hold a CUDA-resident cache
//! from `cuda_backend`.
//!
//! See [`docs/KV.md`](../../../docs/KV.md) for the host vs CUDA design.

use super::DecoderConfig;
#[cfg(feature = "cuda")]
use super::cuda_backend::CudaKVCache;

use crate::{Error, Result};

/// Pluggable past/present KV strategy used by autoregressive decode.
pub(crate) trait KVCacheBackend {
    fn batch_size(&self) -> usize;
    fn past_sequence_length(&self) -> usize;
    fn is_empty(&self) -> bool;
}

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

/// Host-resident decoder key/value cache.
///
/// `KVCache` contains one [`LayerCache`] for each decoder layer. The cache is
/// initialized empty for the first decoder pass. During autoregressive
/// generation the decoder overwrites each layer's buffers in place, reusing
/// allocation capacity across steps.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KVCache {
    layers: Vec<LayerCache>,
    batch_size: usize,
    past_sequence_length: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl KVCache {
    /// Creates an empty KV cache for a decoder configuration and batch size.
    pub fn empty(config: &DecoderConfig, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidKVCache {
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
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidKVCache {
                reason: "batch size must be greater than zero".to_owned(),
            });
        }
        if num_key_value_heads == 0 {
            return Err(Error::InvalidKVCache {
                reason: "KV head count must be greater than zero".to_owned(),
            });
        }
        if head_dim == 0 {
            return Err(Error::InvalidKVCache {
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
                return Err(Error::InvalidKVCache {
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
                return Err(Error::InvalidKVCache {
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

impl KVCacheBackend for KVCache {
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

/// Selected KV backend for one autoregressive generate run.
pub(crate) enum ActiveKVCache {
    Host(KVCache),
    #[cfg(feature = "cuda")]
    Cuda(CudaKVCache),
}

impl KVCacheBackend for ActiveKVCache {
    fn batch_size(&self) -> usize {
        match self {
            Self::Host(cache) => cache.batch_size,
            #[cfg(feature = "cuda")]
            Self::Cuda(cache) => cache.batch_size(),
        }
    }

    fn past_sequence_length(&self) -> usize {
        match self {
            Self::Host(cache) => cache.past_sequence_length,
            #[cfg(feature = "cuda")]
            Self::Cuda(cache) => cache.past_sequence_length(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Host(cache) => cache.past_sequence_length == 0,
            #[cfg(feature = "cuda")]
            Self::Cuda(cache) => cache.is_empty(),
        }
    }
}

pub(crate) fn values_per_tensor(
    batch_size: usize,
    sequence_length: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Result<usize> {
    batch_size
        .checked_mul(num_key_value_heads)
        .and_then(|value| value.checked_mul(sequence_length))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::InvalidKVCache {
            reason: "KV cache tensor shape is too large".to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_kv_cache_backend_surface() {
        let layers = vec![
            LayerCache::new(Vec::new(), Vec::new()),
            LayerCache::new(Vec::new(), Vec::new()),
        ];
        let cache = KVCache::new(layers, 1, 0, 2, 4).unwrap();
        assert!(cache.is_empty());
        assert_eq!(KVCacheBackend::batch_size(&cache), 1);
        assert_eq!(KVCacheBackend::past_sequence_length(&cache), 0);
    }

    #[test]
    fn active_host_backend() {
        let layers = vec![LayerCache::new(Vec::new(), Vec::new())];
        let cache = KVCache::new(layers, 2, 0, 1, 4).unwrap();
        let active = ActiveKVCache::Host(cache);
        assert_eq!(KVCacheBackend::batch_size(&active), 2);
        assert!(KVCacheBackend::is_empty(&active));
    }
}
