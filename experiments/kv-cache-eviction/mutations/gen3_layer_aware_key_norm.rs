//! Evolved eviction scoring function
//!
//! Gen3 Crossover: Layer-aware weighting + Key norm penalty
//! Combines the two best Gen2 mutations:
//! 1. Layer-aware weighting (PyramidKV-inspired)
//! 2. Key norm penalty (KnormPress-inspired)

use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

/// Champion eviction scorer discovered through evolution.
///
/// Gen3 crossover combining:
/// - Layer-aware attention weighting from Gen2 layer_aware
/// - Key norm penalty from Gen2 key_norm
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

        // Layer-aware weighting (from Gen2 layer_aware)
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.7 - 0.2 * layer_ratio;  // 0.7 at layer 0, 0.5 at last layer
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;  // 0.3 at layer 0, 0.5 at last layer

        // Position correction factor
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(0.3);

        // Recency bonus
        let recency_bonus = if token.relative_pos < 128 {
            0.2 * (1.0 - token.relative_pos as f64 / 128.0)
        } else {
            0.0
        };

        // Key norm penalty (from Gen2 key_norm)
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
