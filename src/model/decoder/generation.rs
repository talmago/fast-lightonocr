//! Autoregressive generation utilities.
//!
//! This module contains helper types and algorithms used by the decoder during
//! autoregressive generation. The decoder owns the runtime state (ONNX session,
//! KV-cache, generation configuration) and delegates token-selection logic to
//! these helpers.

use crate::model::Logits;
use crate::model::decoder::GenerationConfig;
use crate::{Error, Result};

/// Reason an autoregressive generation run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Generation produced a configured end-of-sequence token.
    EndOfSequence,

    /// Generation reached the configured maximum number of new tokens.
    Length,
}

/// Output of an autoregressive generation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    token_ids: Vec<i64>,
    finish_reason: FinishReason,
}

impl GenerationOutput {
    /// Creates a generation output.
    #[must_use]
    pub fn new(token_ids: Vec<i64>, finish_reason: FinishReason) -> Self {
        Self {
            token_ids,
            finish_reason,
        }
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
}

/// Selects the next token according to the configured decoding strategy.
pub(super) fn next_token(
    config: &GenerationConfig,
    logits: &Logits,
    rng: &mut fastrand::Rng,
) -> Result<i64> {
    if config.do_sample {
        sample_next_token(config, logits, rng)
    } else {
        greedy_next_token(logits)
    }
}

/// Returns whether the token is an end-of-sequence token.
#[must_use]
pub(super) fn is_eos(config: &GenerationConfig, token_id: i64) -> bool {
    config.eos_token_ids.contains(&token_id)
}

/// Greedily selects the highest-scoring token from the final decoder position.
fn greedy_next_token(logits: &Logits) -> Result<i64> {
    let scores = final_position_scores(logits)?;

    let (token_id, _) = scores
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("validated non-empty vocabulary");

    token_id_to_i64(token_id)
}

fn sample_next_token(
    config: &GenerationConfig,
    logits: &Logits,
    rng: &mut fastrand::Rng,
) -> Result<i64> {
    let scores = final_position_scores(logits)?;

    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(Error::Inference {
            reason: format!(
                "sampling temperature must be finite and greater than zero, found {}",
                config.temperature
            ),
        });
    }

    let mut candidates = scores
        .iter()
        .enumerate()
        .filter_map(|(token_id, &score)| {
            score.is_finite().then_some(Candidate {
                token_id,
                score: score / config.temperature,
            })
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return Err(Error::Inference {
            reason: "sampling requires at least one finite logit".to_owned(),
        });
    }

    apply_top_k(&mut candidates, config.top_k);
    apply_top_p(&mut candidates, config.top_p)?;
    sample_candidate(&candidates, rng)
}

fn final_position_scores(logits: &Logits) -> Result<&[f32]> {
    let (batch_size, sequence_length, vocab_size) = logits.shape();

    if batch_size != 1 {
        return Err(Error::Inference {
            reason: format!("only batch size 1 is supported, found {batch_size}"),
        });
    }

    if sequence_length == 0 {
        return Err(Error::Inference {
            reason: "logits sequence length must be greater than zero".to_owned(),
        });
    }

    if vocab_size == 0 {
        return Err(Error::Inference {
            reason: "logits vocabulary size must be greater than zero".to_owned(),
        });
    }

    let start = (sequence_length - 1) * vocab_size;
    Ok(&logits.as_slice()[start..start + vocab_size])
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    token_id: usize,
    score: f32,
}

fn apply_top_k(candidates: &mut Vec<Candidate>, top_k: u32) {
    let Ok(top_k) = usize::try_from(top_k) else {
        return;
    };
    if top_k == 0 || top_k >= candidates.len() {
        return;
    }

    sort_candidates(candidates);
    candidates.truncate(top_k);
}

fn apply_top_p(candidates: &mut Vec<Candidate>, top_p: f32) -> Result<()> {
    if !top_p.is_finite() {
        return Err(Error::Inference {
            reason: format!("top-p must be finite, found {top_p}"),
        });
    }
    if top_p >= 1.0 {
        return Ok(());
    }
    if top_p <= 0.0 {
        return Err(Error::Inference {
            reason: format!("top-p must be greater than zero, found {top_p}"),
        });
    }

    sort_candidates(candidates);

    let probabilities = softmax_probabilities(candidates)?;
    let mut cumulative_probability = 0.0;
    let mut keep_count = 0;
    for probability in probabilities {
        cumulative_probability += probability;
        keep_count += 1;
        if cumulative_probability >= top_p {
            break;
        }
    }

    candidates.truncate(keep_count.max(1));
    Ok(())
}

fn sample_candidate(candidates: &[Candidate], rng: &mut fastrand::Rng) -> Result<i64> {
    let probabilities = softmax_probabilities(candidates)?;
    let draw = rng.f32();
    let mut cumulative_probability = 0.0;

    for (candidate, probability) in candidates.iter().zip(probabilities) {
        cumulative_probability += probability;
        if draw < cumulative_probability {
            return token_id_to_i64(candidate.token_id);
        }
    }

    let candidate = candidates.last().ok_or_else(|| Error::Inference {
        reason: "sampling requires at least one candidate".to_owned(),
    })?;
    token_id_to_i64(candidate.token_id)
}

fn softmax_probabilities(candidates: &[Candidate]) -> Result<Vec<f32>> {
    if candidates.is_empty() {
        return Err(Error::Inference {
            reason: "sampling requires at least one candidate".to_owned(),
        });
    }

    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score)
        .max_by(f32::total_cmp)
        .expect("validated non-empty candidates");

    let weights = candidates
        .iter()
        .map(|candidate| (candidate.score - max_score).exp())
        .collect::<Vec<_>>();
    let total_weight = weights.iter().copied().sum::<f32>();

    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(Error::Inference {
            reason: "sampling probabilities could not be normalized".to_owned(),
        });
    }

    Ok(weights
        .into_iter()
        .map(|weight| weight / total_weight)
        .collect())
}

fn sort_candidates(candidates: &mut [Candidate]) {
    candidates.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

fn token_id_to_i64(token_id: usize) -> Result<i64> {
    i64::try_from(token_id).map_err(|_| Error::Inference {
        reason: format!("selected token id {token_id} does not fit in i64"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation_config() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 8,
            bos_token_id: 151_643,
            do_sample: false,
            eos_token_ids: vec![151_645, 151_643],
            pad_token_id: 151_643,
            temperature: 0.2,
            top_k: 0,
            top_p: 0.9,
            transformers_version: None,
            trust_remote_code: false,
        }
    }

    #[test]
    fn greedy_next_token_selects_argmax_from_final_position() {
        let logits = Logits::new(vec![10.0, 0.0, 1.0, 0.1, 0.2, 0.3], 1, 2, 3).unwrap();

        assert_eq!(greedy_next_token(&logits).unwrap(), 2);
    }

    #[test]
    fn next_token_uses_greedy_when_sampling_is_disabled() {
        let mut config = generation_config();
        config.do_sample = false;
        config.temperature = 0.001;
        config.top_k = 1;
        config.top_p = 0.001;
        let logits = Logits::new(vec![0.0, 2.0, 1.0], 1, 1, 3).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);

        assert_eq!(next_token(&config, &logits, &mut rng).unwrap(), 1);
    }

    #[test]
    fn sampled_next_token_honors_top_k_filter() {
        let mut config = generation_config();
        config.do_sample = true;
        config.temperature = 1.0;
        config.top_k = 1;
        config.top_p = 1.0;
        let logits = Logits::new(vec![0.0, 3.0, 1.0], 1, 1, 3).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);

        assert_eq!(next_token(&config, &logits, &mut rng).unwrap(), 1);
    }

    #[test]
    fn sampled_next_token_honors_top_p_filter() {
        let mut config = generation_config();
        config.do_sample = true;
        config.temperature = 1.0;
        config.top_k = 0;
        config.top_p = 0.5;
        let logits = Logits::new(vec![3.0, 2.0, 1.0], 1, 1, 3).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);

        assert_eq!(next_token(&config, &logits, &mut rng).unwrap(), 0);
    }

    #[test]
    fn greedy_next_token_rejects_empty_sequence() {
        let logits = Logits::new(Vec::new(), 1, 0, 3).unwrap();

        let error = greedy_next_token(&logits).expect_err("empty sequence should fail");

        match error {
            Error::Inference { reason } => {
                assert!(reason.contains("sequence length"));
            }
            other => panic!("expected inference error, got {other:?}"),
        }
    }

    #[test]
    fn eos_detection_matches_any_configured_eos_token() {
        let config = generation_config();

        assert!(is_eos(&config, 151_645));
        assert!(is_eos(&config, 151_643));
        assert!(!is_eos(&config, 42));
    }

    #[test]
    fn generation_output_exposes_finish_reason() {
        let output = GenerationOutput::new(vec![1, 2], FinishReason::EndOfSequence);

        assert_eq!(output.token_ids(), &[1, 2]);
        assert_eq!(output.finish_reason(), FinishReason::EndOfSequence);
    }
}
