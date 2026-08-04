//! Autoregressive generation utilities.
//!
//! This module contains helper types and algorithms used by the decoder during
//! autoregressive generation. The decoder owns the runtime state (ONNX session,
//! KV-cache, generation configuration) and delegates token-selection logic to
//! these helpers.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{Error, Result};

use super::config::GenerationConfig;
use super::logits::Logits;

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

    let mut candidates = Vec::with_capacity(scores.len());
    for (token_id, &score) in scores.iter().enumerate() {
        if score.is_finite() {
            candidates.push(Candidate {
                token_id,
                score: score / config.temperature,
            });
        }
    }

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

/// Keeps the `top_k` highest-scoring candidates without sorting the full list.
///
/// Uses `select_nth_unstable_by` so cost is linearithmic in the worst case for
/// selection, then truncates. The retained prefix is unsorted.
fn apply_top_k(candidates: &mut Vec<Candidate>, top_k: u32) {
    let Ok(top_k) = usize::try_from(top_k) else {
        return;
    };
    if top_k == 0 || top_k >= candidates.len() {
        return;
    }

    candidates.select_nth_unstable_by(top_k - 1, compare_candidates_descending);
    candidates.truncate(top_k);
}

/// Truncates to the nucleus of candidates whose cumulative softmax mass reaches
/// `top_p`, without sorting the full candidate list.
///
/// Builds an O(n) max-heap over scores and extracts highest-scoring tokens until
/// the cumulative probability is at least `top_p`. For peaked distributions
/// (typical with low temperature) the nucleus is much smaller than the vocab,
/// so this is far cheaper than an O(n log n) full sort.
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

    let probabilities = softmax_probabilities(candidates)?;

    // Small candidate sets (typical after top-k) are cheaper to sort directly.
    // Large sets use a max-heap so we only extract the nucleus, not sort all.
    if candidates.len() <= 64 {
        let mut order = (0..candidates.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|&left, &right| {
            compare_candidates_descending(&candidates[left], &candidates[right])
        });

        let mut selected = Vec::new();
        let mut cumulative_probability = 0.0;
        for index in order {
            selected.push(candidates[index]);
            cumulative_probability += probabilities[index];
            if cumulative_probability >= top_p {
                break;
            }
        }
        *candidates = selected;
        return Ok(());
    }

    let heap = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| NucleusItem {
            score: candidate.score,
            token_id: candidate.token_id,
            index,
        })
        .collect::<BinaryHeap<_>>();

    let mut selected = Vec::new();
    let mut cumulative_probability = 0.0;
    let mut heap = heap;
    while let Some(item) = heap.pop() {
        selected.push(candidates[item.index]);
        cumulative_probability += probabilities[item.index];
        if cumulative_probability >= top_p {
            break;
        }
    }

    if selected.is_empty() {
        return Err(Error::Inference {
            reason: "top-p filtering removed every candidate".to_owned(),
        });
    }

    *candidates = selected;
    Ok(())
}

/// Max-heap item: higher score first; lower token id wins ties (matches prior sort).
#[derive(Clone, Copy, Debug)]
struct NucleusItem {
    score: f32,
    token_id: usize,
    index: usize,
}

impl PartialEq for NucleusItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for NucleusItem {}

impl PartialOrd for NucleusItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NucleusItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.token_id.cmp(&self.token_id))
    }
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

fn compare_candidates_descending(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.token_id.cmp(&right.token_id))
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

    fn sort_based_top_p(candidates: &mut Vec<Candidate>, top_p: f32) {
        candidates.sort_unstable_by(compare_candidates_descending);
        let probabilities = softmax_probabilities(candidates).unwrap();
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
    fn top_p_nucleus_matches_sort_based_selection() {
        let scores = [
            3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1, -1.0, -2.0, 4.0, 0.0,
        ];
        assert_top_p_matches_sort(&scores, 0.9);
    }

    #[test]
    fn top_p_nucleus_matches_sort_based_selection_large_vocab() {
        // Exercises the heap path (candidate count > 64).
        let mut scores = vec![0.0; 512];
        for (index, score) in scores.iter_mut().enumerate() {
            *score = (index as f32 * 0.037) % 5.0 - 1.0;
        }
        scores[17] = 8.0;
        scores[99] = 7.5;
        scores[250] = 7.0;
        assert_top_p_matches_sort(&scores, 0.9);
    }

    fn assert_top_p_matches_sort(scores: &[f32], top_p: f32) {
        let mut heap_based = scores
            .iter()
            .enumerate()
            .map(|(token_id, &score)| Candidate { token_id, score })
            .collect::<Vec<_>>();
        let mut sort_based = heap_based.clone();

        apply_top_p(&mut heap_based, top_p).unwrap();
        sort_based_top_p(&mut sort_based, top_p);

        let mut heap_ids = heap_based
            .iter()
            .map(|candidate| candidate.token_id)
            .collect::<Vec<_>>();
        let mut sort_ids = sort_based
            .iter()
            .map(|candidate| candidate.token_id)
            .collect::<Vec<_>>();
        heap_ids.sort_unstable();
        sort_ids.sort_unstable();
        assert_eq!(heap_ids, sort_ids);
    }

    #[test]
    fn top_k_select_keeps_highest_scoring_tokens() {
        let mut candidates = vec![
            Candidate {
                token_id: 0,
                score: 1.0,
            },
            Candidate {
                token_id: 1,
                score: 5.0,
            },
            Candidate {
                token_id: 2,
                score: 3.0,
            },
            Candidate {
                token_id: 3,
                score: 4.0,
            },
            Candidate {
                token_id: 4,
                score: 2.0,
            },
        ];

        apply_top_k(&mut candidates, 3);

        let mut ids = candidates
            .iter()
            .map(|candidate| candidate.token_id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3]);
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
