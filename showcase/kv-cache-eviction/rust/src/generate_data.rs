//! Synthetic Attention Pattern Generator
//!
//! Generates realistic attention patterns for benchmarking eviction policies.
//! Patterns are designed to mimic known LLM attention behaviors:
//! - Attention sinks (first tokens attract high attention)
//! - Recency bias (recent tokens get more attention)
//! - Information-dense tokens (some tokens consistently important)
//! - Layer-dependent patterns (early layers more diffuse, late layers more focused)

use rand::prelude::*;
use kv_cache::AttentionPattern;

/// Configuration for synthetic pattern generation
struct PatternConfig {
    seq_len: usize,
    num_layers: usize,
    /// Number of sink tokens (typically 4)
    num_sinks: usize,
    /// Strength of sink attention (0.0-1.0)
    sink_strength: f64,
    /// Recency window for local attention
    local_window: usize,
    /// Strength of local/recent attention
    local_strength: f64,
    /// Fraction of tokens that are "information dense"
    info_dense_fraction: f64,
    /// How much more attention info-dense tokens get
    info_dense_multiplier: f64,
    /// Random seed for reproducibility
    seed: u64,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            num_layers: 32,
            num_sinks: 4,
            sink_strength: 0.15,
            local_window: 64,
            local_strength: 0.4,
            info_dense_fraction: 0.1,
            info_dense_multiplier: 3.0,
            seed: 42,
        }
    }
}

/// Generate a single attention pattern
fn generate_pattern(config: &PatternConfig, pattern_seed: u64) -> AttentionPattern {
    let mut rng = StdRng::seed_from_u64(config.seed ^ pattern_seed);

    let seq_len = config.seq_len;
    let num_layers = config.num_layers;

    // Determine which tokens are "information dense"
    let num_info_dense = (seq_len as f64 * config.info_dense_fraction) as usize;
    let mut info_dense_positions: Vec<usize> = (config.num_sinks..seq_len).collect();
    info_dense_positions.shuffle(&mut rng);
    let info_dense_positions: std::collections::HashSet<usize> =
        info_dense_positions.into_iter().take(num_info_dense).collect();

    // Generate attention patterns for each layer
    let mut attention = vec![vec![vec![0.0; seq_len]; seq_len]; num_layers];
    let mut key_norms = vec![vec![0.0; seq_len]; num_layers];

    for layer in 0..num_layers {
        // Layer-dependent focus: later layers are more focused
        let layer_ratio = layer as f64 / num_layers as f64;
        let focus_factor = 1.0 + 2.0 * layer_ratio; // Later layers more peaked

        for query in 0..seq_len {
            let mut raw_attn = vec![0.0; seq_len];

            for key in 0..=query {
                // Start with uniform base
                let mut weight = 1.0;

                // Sink attention
                if key < config.num_sinks {
                    weight += config.sink_strength * (config.num_sinks - key) as f64;
                }

                // Local/recency attention
                let distance = query - key;
                if distance < config.local_window {
                    weight += config.local_strength
                        * (1.0 - distance as f64 / config.local_window as f64);
                }

                // Information-dense token bonus
                if info_dense_positions.contains(&key) {
                    weight *= config.info_dense_multiplier;
                }

                // Position-based decay (long-range attention weaker)
                let position_decay = 1.0 / (1.0 + 0.001 * distance as f64);
                weight *= position_decay;

                // Layer-dependent sharpening
                weight = weight.powf(focus_factor);

                // Add some noise
                weight *= 0.8 + 0.4 * rng.gen::<f64>();

                raw_attn[key] = weight;
            }

            // Softmax normalization (only over valid keys 0..=query)
            let max_val = raw_attn[0..=query]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = raw_attn[0..=query]
                .iter()
                .map(|&x| (x - max_val).exp())
                .sum();

            for key in 0..=query {
                attention[layer][query][key] = ((raw_attn[key] - max_val).exp()) / sum;
            }
        }

        // Generate key norms
        // Sink tokens and info-dense tokens typically have lower norms
        // Outlier tokens have high norms
        for pos in 0..seq_len {
            let mut norm = 1.0;

            if pos < config.num_sinks {
                norm *= 0.7; // Sink tokens: lower norm
            }

            if info_dense_positions.contains(&pos) {
                norm *= 0.8; // Important tokens: lower norm
            }

            // Some random outliers with high norms
            if rng.gen::<f64>() < 0.05 {
                norm *= 2.0 + rng.gen::<f64>() * 2.0;
            }

            // Add noise
            norm *= 0.9 + 0.2 * rng.gen::<f64>();

            key_norms[layer][pos] = norm;
        }
    }

    // Token types: mostly regular (0), some special (1), some punctuation (2)
    let token_types: Vec<u8> = (0..seq_len)
        .map(|_| {
            let r: f64 = rng.gen();
            if r < 0.85 {
                0
            } else if r < 0.95 {
                1
            } else {
                2
            }
        })
        .collect();

    AttentionPattern {
        seq_len,
        num_layers,
        attention,
        key_norms,
        token_types,
        important_positions: info_dense_positions.into_iter().collect(),
    }
}

/// Generate a set of patterns for TRAIN/VALID/TEST split
fn generate_dataset(num_train: usize, num_valid: usize, num_test: usize) -> (Vec<AttentionPattern>, Vec<AttentionPattern>, Vec<AttentionPattern>) {
    let base_config = PatternConfig::default();

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();

    // Training patterns with varied configurations
    for i in 0..num_train {
        let mut config = base_config.clone();
        config.seed = 1000 + i as u64;
        // Vary parameters for diversity
        config.sink_strength = 0.1 + 0.1 * (i % 5) as f64 / 4.0;
        config.local_strength = 0.3 + 0.2 * (i % 4) as f64 / 3.0;
        config.seq_len = 256 + 128 * (i % 4);
        train.push(generate_pattern(&config, i as u64));
    }

    // Validation patterns
    for i in 0..num_valid {
        let mut config = base_config.clone();
        config.seed = 2000 + i as u64;
        config.seq_len = 384 + 64 * (i % 3);
        valid.push(generate_pattern(&config, 1000 + i as u64));
    }

    // Test patterns (different distribution)
    for i in 0..num_test {
        let mut config = base_config.clone();
        config.seed = 3000 + i as u64;
        config.seq_len = 512 + 256 * (i % 3);
        config.num_layers = 24 + 8 * (i % 2); // Different model sizes
        test.push(generate_pattern(&config, 2000 + i as u64));
    }

    (train, valid, test)
}

fn main() {
    println!("Generating synthetic attention patterns...");

    let (train, valid, test) = generate_dataset(20, 10, 10);

    println!("Generated:");
    println!("  TRAIN: {} patterns", train.len());
    println!("  VALID: {} patterns", valid.len());
    println!("  TEST:  {} patterns", test.len());

    // Print some statistics
    let avg_train_len: f64 = train.iter().map(|p| p.seq_len as f64).sum::<f64>() / train.len() as f64;
    let avg_valid_len: f64 = valid.iter().map(|p| p.seq_len as f64).sum::<f64>() / valid.len() as f64;
    let avg_test_len: f64 = test.iter().map(|p| p.seq_len as f64).sum::<f64>() / test.len() as f64;

    println!("\nAverage sequence lengths:");
    println!("  TRAIN: {:.0}", avg_train_len);
    println!("  VALID: {:.0}", avg_valid_len);
    println!("  TEST:  {:.0}", avg_test_len);

    // Quick sanity check: verify attention sums to ~1.0
    let pattern = &train[0];
    let layer = 0;
    let query = pattern.seq_len - 1;
    let attn_sum: f64 = pattern.attention[layer][query].iter().sum();
    println!("\nSanity check - attention sum at last position: {:.6}", attn_sum);

    // Verify sink tokens have high attention
    let sink_attn: f64 = pattern.attention[layer][query][0..4].iter().sum();
    println!("Attention to sink tokens (0-3): {:.4}", sink_attn);

    println!("\nData generation complete. Ready for evolution.");
}

impl Clone for PatternConfig {
    fn clone(&self) -> Self {
        Self {
            seq_len: self.seq_len,
            num_layers: self.num_layers,
            num_sinks: self.num_sinks,
            sink_strength: self.sink_strength,
            local_window: self.local_window,
            local_strength: self.local_strength,
            info_dense_fraction: self.info_dense_fraction,
            info_dense_multiplier: self.info_dense_multiplier,
            seed: self.seed,
        }
    }
}

/// Public function to generate patterns for benchmarking
pub fn generate_benchmark_patterns() -> (Vec<AttentionPattern>, Vec<AttentionPattern>, Vec<AttentionPattern>) {
    generate_dataset(20, 10, 10)
}
