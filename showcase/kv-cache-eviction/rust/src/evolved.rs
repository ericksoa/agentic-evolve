//! Evolved eviction scoring function
//!
//! Champion KV-cache eviction scorer discovered through evolution.
//! Achieves 7.07% TRAIN / 6.87% VALID / 6.85% TEST improvement over hybrid baseline.
//!
//! Key innovations:
//! 1. Layer-aware attention weighting (PyramidKV-inspired)
//! 2. Key norm penalty for outliers (KnormPress-inspired)
//! 3. Layer-adaptive position correction
//! 4. Layer-aware recency bonus

use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

/// Champion eviction scorer discovered through evolution.
///
/// Gen4 Champion combining:
/// - Layer-aware attention weights (0.7→0.5 recent, 0.3→0.5 cumulative)
/// - Key norm penalty for outlier tokens
/// - Layer-adaptive position power (0.25→0.35)
/// - Layer-aware recency bonus (0.15→0.25)
impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens always kept (attention sink phenomenon)
        if token.is_sink {
            return f64::MAX;
        }

        // Very recent tokens always kept
        if token.relative_pos < 4 {
            return 1e6 - token.relative_pos as f64;
        }

        // Layer-aware weighting inspired by PyramidKV
        // Early layers have more diverse attention patterns -> favor recency more
        // Later layers have focused attention -> trust cumulative more
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.7 - 0.2 * layer_ratio;  // 0.7 at layer 0, 0.5 at last layer
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;  // 0.3 at layer 0, 0.5 at last layer

        // Layer-adaptive position correction (stronger for late layers)
        let position_power = 0.25 + 0.1 * layer_ratio;  // 0.25 early, 0.35 late
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Layer-aware recency bonus (stronger for late layers)
        let recency_window = 128;
        let base_recency = 0.15 + 0.1 * layer_ratio;  // 0.15 early, 0.25 late
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else {
            0.0
        };

        // Key norm penalty (from Gen2 key_norm - KnormPress-inspired)
        // Tokens with large key norms (outliers) are penalized
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // Combined score
        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "evolved"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolved_sink_priority() {
        let scorer = Evolved;

        let mut sink = TokenInfo::new(2, 100);
        sink.is_sink = true;
        assert_eq!(scorer.score(&sink), f64::MAX);
    }

    #[test]
    fn test_evolved_attention_sensitivity() {
        let scorer = Evolved;

        let mut high_attn = TokenInfo::new(50, 200);
        high_attn.recent_attn = 5.0;
        high_attn.cumulative_attn = 10.0;

        let mut low_attn = TokenInfo::new(50, 200);
        low_attn.recent_attn = 0.5;
        low_attn.cumulative_attn = 1.0;

        assert!(scorer.score(&high_attn) > scorer.score(&low_attn));
    }
}
