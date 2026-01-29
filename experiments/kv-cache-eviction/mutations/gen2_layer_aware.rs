//! Evolved eviction scoring function
//!
//! This module contains the champion scoring function discovered through evolution.
//! Gen2 mutation: Layer-aware weighting inspired by PyramidKV
//! - Early layers have more diverse attention patterns -> favor recency more
//! - Later layers have focused attention -> trust cumulative more

use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

/// Champion eviction scorer discovered through evolution.
///
/// Gen2: Layer-aware weighting
/// - Early layers: recent_weight=0.7, cumulative_weight=0.3
/// - Late layers: recent_weight=0.5, cumulative_weight=0.5
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

        // Position correction factor to counter H2O's bias
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(0.3);

        // Recency bonus for somewhat recent tokens
        let recency_bonus = if token.relative_pos < 128 {
            0.2 * (1.0 - token.relative_pos as f64 / 128.0)
        } else {
            0.0
        };

        // Combined score
        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
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
