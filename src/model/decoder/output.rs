//! Decoder output value.

use super::kv_cache::KVCache;
use super::logits::Logits;

/// Output from a single decoder invocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecoderOutput {
    /// Logits for the sequence positions processed by the decoder pass.
    pub logits: Logits,

    /// Updated key/value cache returned by the decoder.
    pub kv_cache: KVCache,
}

impl DecoderOutput {
    /// Creates decoder output from logits and an updated KV cache.
    pub fn new(logits: Logits, kv_cache: KVCache) -> Self {
        Self { logits, kv_cache }
    }
}
