//! Evolved eviction scoring function
//!
//! Gen3 Full Crossover: Layer-aware + Key norm + Position power tuning
//! Combines all promising Gen2 mutations

use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

/// Champion eviction scorer discovered through evolution.
///
/// Gen3 full crossover combining:
/// - Layer-aware attention weighting
/// - Key norm penalty
/// - Layer-adaptive position power (0.25 early, 0.35 late)
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

        // Layer-aware weighting
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.7 - 0.2 * layer_ratio;
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;

        // Layer-adaptive position correction (stronger for late layers)
        let position_power = 0.25 + 0.1 * layer_ratio;  // 0.25 early, 0.35 late
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Recency bonus
        let recency_bonus = if token.relative_pos < 128 {
            0.2 * (1.0 - token.relative_pos as f64 / 128.0)
        } else {
            0.0
        };

        // Key norm penalty
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
