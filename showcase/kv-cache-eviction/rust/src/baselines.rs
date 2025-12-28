//! Baseline eviction scoring methods
//!
//! Implementations of known KV-cache eviction policies for comparison.

use crate::{EvictionScorer, TokenInfo};

/// StreamingLLM: Keep sink tokens (first N) + recent window
///
/// Reference: Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks"
///
/// Simple position-based policy:
/// - Always keep first `num_sinks` tokens (attention sinks)
/// - Keep `window_size` most recent tokens
/// - Evict everything in between
pub struct StreamingLLM {
    num_sinks: usize,
    window_size: usize,
}

impl StreamingLLM {
    pub fn new(num_sinks: usize, window_size: usize) -> Self {
        Self { num_sinks, window_size }
    }
}

impl EvictionScorer for StreamingLLM {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens get maximum priority
        if token.position < self.num_sinks {
            return f64::MAX;
        }

        // Recent tokens get high priority based on recency
        if token.relative_pos < self.window_size {
            return 1e6 - token.relative_pos as f64;
        }

        // Everything else gets evicted
        f64::NEG_INFINITY
    }

    fn name(&self) -> &'static str {
        "streaming_llm"
    }
}

/// H2O: Heavy-Hitter Oracle
///
/// Reference: Zhang et al. (2024) "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"
///
/// Keep tokens with highest cumulative attention scores + recent tokens.
/// Known weakness: Position bias - tokens past position 1500 get disproportionately evicted.
pub struct H2O {
    recent_window: usize,
}

impl H2O {
    pub fn new(recent_window: usize) -> Self {
        Self { recent_window }
    }
}

impl EvictionScorer for H2O {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Recent tokens always kept
        if token.relative_pos < self.recent_window {
            return f64::MAX;
        }

        // Otherwise, score by cumulative attention
        token.cumulative_attn
    }

    fn name(&self) -> &'static str {
        "h2o"
    }
}

/// SnapKV-lite: Recent attention as importance proxy
///
/// Reference: Li et al. (2024) "SnapKV: LLM Knows What You are Looking for Before Generation"
///
/// Simplified version using recent attention in observation window.
/// Full SnapKV uses clustering which is more complex.
pub struct SnapKVLite {
    observation_window: usize,
}

impl SnapKVLite {
    pub fn new(observation_window: usize) -> Self {
        Self { observation_window }
    }
}

impl EvictionScorer for SnapKVLite {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Recent tokens always kept
        if token.relative_pos < self.observation_window {
            return f64::MAX;
        }

        // Score by recent attention (attention in last K steps)
        token.recent_attn
    }

    fn name(&self) -> &'static str {
        "snapkv_lite"
    }
}

/// KnormPress: Evict tokens with large key L2 norms
///
/// Reference: NVIDIA kvpress "KnormPress"
///
/// Tokens with large key norms are often outliers that don't contribute
/// meaningfully to attention. Evict them first.
pub struct KnormPress;

impl EvictionScorer for KnormPress {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Negative key norm: larger norm = lower score = evict first
        -token.key_norm
    }

    fn name(&self) -> &'static str {
        "knorm_press"
    }
}

/// TOVA: Token-wise Observed Value with Attention
///
/// Reference: Based on attention tracking methods
///
/// Combines cumulative attention with recency weighting.
pub struct TOVA {
    recency_decay: f64,
}

impl TOVA {
    pub fn new(recency_decay: f64) -> Self {
        Self { recency_decay }
    }
}

impl EvictionScorer for TOVA {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens get bonus
        let sink_bonus = if token.is_sink { 10.0 } else { 0.0 };

        // Recency weight
        let recency_weight = (-self.recency_decay * token.relative_pos as f64).exp();

        // Combined score
        sink_bonus + token.cumulative_attn * (1.0 + recency_weight)
    }

    fn name(&self) -> &'static str {
        "tova"
    }
}

/// PyramidKV-inspired: Layer-aware scoring
///
/// Reference: Cai et al. (2024) "PyramidKV: Dynamic KV Cache Compression"
///
/// Different layers should have different cache budgets.
/// Early layers: more diverse attention, need more tokens
/// Later layers: more focused attention, can use fewer tokens
pub struct PyramidKV {
    base_recent_window: usize,
}

impl PyramidKV {
    pub fn new(base_recent_window: usize) -> Self {
        Self { base_recent_window }
    }
}

impl EvictionScorer for PyramidKV {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Layer-dependent recency bonus
        // Early layers get larger effective windows
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let effective_window = self.base_recent_window as f64 * (2.0 - layer_ratio);

        // Sink tokens always high priority
        if token.is_sink {
            return f64::MAX;
        }

        // Score based on relative position with layer adjustment
        if (token.relative_pos as f64) < effective_window {
            return 1e6 - token.relative_pos as f64;
        }

        // Non-recent: score by attention
        token.cumulative_attn * (1.0 - 0.5 * layer_ratio)
    }

    fn name(&self) -> &'static str {
        "pyramid_kv"
    }
}

/// Hybrid baseline combining multiple signals
///
/// Combines attention scores, recency, key norms, and sink detection.
pub struct HybridBaseline {
    pub attn_weight: f64,
    pub recency_weight: f64,
    pub recent_window: usize,
}

impl HybridBaseline {
    pub fn new() -> Self {
        Self {
            attn_weight: 0.4,
            recency_weight: 0.3,
            recent_window: 64,
        }
    }
}

impl Default for HybridBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionScorer for HybridBaseline {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens always kept
        if token.is_sink {
            return f64::MAX;
        }

        // Recent tokens get high score
        let recency_score = if token.relative_pos < self.recent_window {
            1.0 - (token.relative_pos as f64 / self.recent_window as f64)
        } else {
            0.0
        };

        // Attention-based score (combine cumulative and recent)
        let attn_score = 0.3 * token.cumulative_attn + 0.7 * token.recent_attn;

        // Penalize large key norms slightly
        let norm_penalty = -0.1 * token.key_norm;

        // Combined weighted score
        self.attn_weight * attn_score
            + self.recency_weight * recency_score
            + norm_penalty
    }

    fn name(&self) -> &'static str {
        "hybrid_baseline"
    }
}

/// Random eviction (control baseline)
///
/// Randomly scores tokens. Should be the worst baseline.
pub struct RandomEviction {
    seed: u64,
}

impl RandomEviction {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl EvictionScorer for RandomEviction {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens always kept
        if token.is_sink {
            return f64::MAX;
        }

        // Very recent tokens kept
        if token.relative_pos < 4 {
            return 1e6 - token.relative_pos as f64;
        }

        // Pseudo-random score based on position
        let state = self.seed.wrapping_mul(6364136223846793005)
            .wrapping_add(token.position as u64);
        (state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn name(&self) -> &'static str {
        "random"
    }
}

/// Position-Corrected H2O
///
/// Addresses H2O's position bias by normalizing attention by position.
pub struct PositionCorrectedH2O {
    recent_window: usize,
    position_power: f64,
}

impl PositionCorrectedH2O {
    pub fn new(recent_window: usize, position_power: f64) -> Self {
        Self { recent_window, position_power }
    }
}

impl EvictionScorer for PositionCorrectedH2O {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Recent tokens always kept
        if token.relative_pos < self.recent_window {
            return f64::MAX;
        }

        // Sink tokens get bonus
        if token.is_sink {
            return f64::MAX;
        }

        // Position-corrected attention
        // Multiply by position^power to counter the position bias
        let position_factor = (token.position as f64 + 1.0).powf(self.position_power);
        token.cumulative_attn * position_factor
    }

    fn name(&self) -> &'static str {
        "position_corrected_h2o"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_llm() {
        let scorer = StreamingLLM::new(4, 64);

        // Sink token
        let mut sink = TokenInfo::new(2, 100);
        sink.is_sink = true;
        assert_eq!(scorer.score(&sink), f64::MAX);

        // Recent token
        let recent = TokenInfo::new(95, 100);
        assert!(scorer.score(&recent) > 0.0);

        // Old token
        let old = TokenInfo::new(20, 100);
        assert_eq!(scorer.score(&old), f64::NEG_INFINITY);
    }

    #[test]
    fn test_h2o() {
        let scorer = H2O::new(32);

        let mut high_attn = TokenInfo::new(50, 100);
        high_attn.cumulative_attn = 10.0;

        let mut low_attn = TokenInfo::new(50, 100);
        low_attn.cumulative_attn = 1.0;

        assert!(scorer.score(&high_attn) > scorer.score(&low_attn));
    }

    #[test]
    fn test_knorm_press() {
        let scorer = KnormPress;

        let mut low_norm = TokenInfo::new(50, 100);
        low_norm.key_norm = 0.5;

        let mut high_norm = TokenInfo::new(50, 100);
        high_norm.key_norm = 2.0;

        // Lower norm = higher score = keep
        assert!(scorer.score(&low_norm) > scorer.score(&high_norm));
    }
}
