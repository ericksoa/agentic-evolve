//! Micro benchmark with tiny patterns for fast iteration
//! Uses seq_len=64, num_layers=4 for near-instant results

use rand::prelude::*;
use kv_cache::{
    baselines::HybridBaseline,
    evolved::Evolved,
    AttentionPattern, EvictionScorer, TokenInfo,
    evaluate_eviction,
};

/// Generate a tiny pattern for fast testing
fn generate_tiny_pattern(seq_len: usize, num_layers: usize, seed: u64) -> AttentionPattern {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut attention = vec![vec![vec![0.0; seq_len]; seq_len]; num_layers];
    let mut key_norms = vec![vec![1.0; seq_len]; num_layers];

    for layer in 0..num_layers {
        let layer_ratio = layer as f64 / num_layers as f64;
        let focus_factor = 1.0 + 2.0 * layer_ratio;

        for query in 0..seq_len {
            let mut raw_attn = vec![0.0; seq_len];

            for key in 0..=query {
                let mut weight = 1.0;

                // Sink attention
                if key < 4 {
                    weight += 0.15 * (4 - key) as f64;
                }

                // Local attention
                let distance = query - key;
                if distance < 32 {
                    weight += 0.4 * (1.0 - distance as f64 / 32.0);
                }

                // Random noise
                weight *= 0.8 + 0.4 * rng.gen::<f64>();
                raw_attn[key] = weight.powf(focus_factor);
            }

            // Softmax
            let max_val = raw_attn[0..=query].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = raw_attn[0..=query].iter().map(|&x| (x - max_val).exp()).sum();

            for key in 0..=query {
                attention[layer][query][key] = ((raw_attn[key] - max_val).exp()) / sum;
            }
        }

        // Key norms with some outliers
        for pos in 0..seq_len {
            key_norms[layer][pos] = if rng.gen::<f64>() < 0.05 {
                2.0 + rng.gen::<f64>() * 2.0
            } else {
                0.9 + 0.2 * rng.gen::<f64>()
            };
        }
    }

    let token_types = (0..seq_len).map(|_| if rng.gen::<f64>() < 0.85 { 0 } else { 1 }).collect();

    AttentionPattern {
        seq_len,
        num_layers,
        attention,
        key_norms,
        token_types,
        important_positions: vec![],
    }
}

fn main() {
    println!("Micro Benchmark (tiny patterns for fast iteration)");
    println!("===================================================\n");

    // Generate tiny patterns - much faster than full benchmark
    let seq_len = 64;
    let num_layers = 4;
    let num_patterns = 5;

    println!("Pattern size: seq_len={}, num_layers={}", seq_len, num_layers);
    println!("Generating {} patterns...\n", num_patterns * 3);

    let train: Vec<_> = (0..num_patterns).map(|i| generate_tiny_pattern(seq_len, num_layers, 1000 + i as u64)).collect();
    let valid: Vec<_> = (0..num_patterns).map(|i| generate_tiny_pattern(seq_len, num_layers, 2000 + i as u64)).collect();
    let test: Vec<_> = (0..num_patterns).map(|i| generate_tiny_pattern(seq_len, num_layers, 3000 + i as u64)).collect();

    let hybrid = HybridBaseline::new();
    let evolved = Evolved;

    println!("Running benchmarks...\n");

    // Evaluate at each compression ratio
    let ratios = [0.25, 0.50, 0.75];

    for (split_name, patterns) in [("TRAIN", &train), ("VALID", &valid), ("TEST", &test)] {
        println!("{}:", split_name);
        println!("{:<20} {:>10} {:>10} {:>10} {:>10}", "Scorer", "25%", "50%", "75%", "avg");
        println!("{}", "-".repeat(60));

        for (name, scorer) in [("hybrid", &hybrid as &dyn EvictionScorer), ("evolved", &evolved as &dyn EvictionScorer)] {
            let mut errors = Vec::new();
            for &ratio in &ratios {
                let mut total = 0.0;
                for pattern in patterns.iter() {
                    total += evaluate_eviction(pattern, scorer, ratio);
                }
                errors.push(total / patterns.len() as f64);
            }
            let avg = errors.iter().sum::<f64>() / errors.len() as f64;
            println!("{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4}", name, errors[0], errors[1], errors[2], avg);
        }
        println!();
    }

    // Calculate improvement
    let hybrid_train: f64 = ratios.iter().map(|&r| {
        train.iter().map(|p| evaluate_eviction(p, &hybrid, r)).sum::<f64>() / train.len() as f64
    }).sum::<f64>() / 3.0;

    let evolved_train: f64 = ratios.iter().map(|&r| {
        train.iter().map(|p| evaluate_eviction(p, &evolved, r)).sum::<f64>() / train.len() as f64
    }).sum::<f64>() / 3.0;

    let hybrid_valid: f64 = ratios.iter().map(|&r| {
        valid.iter().map(|p| evaluate_eviction(p, &hybrid, r)).sum::<f64>() / valid.len() as f64
    }).sum::<f64>() / 3.0;

    let evolved_valid: f64 = ratios.iter().map(|&r| {
        valid.iter().map(|p| evaluate_eviction(p, &evolved, r)).sum::<f64>() / valid.len() as f64
    }).sum::<f64>() / 3.0;

    let hybrid_test: f64 = ratios.iter().map(|&r| {
        test.iter().map(|p| evaluate_eviction(p, &hybrid, r)).sum::<f64>() / test.len() as f64
    }).sum::<f64>() / 3.0;

    let evolved_test: f64 = ratios.iter().map(|&r| {
        test.iter().map(|p| evaluate_eviction(p, &evolved, r)).sum::<f64>() / test.len() as f64
    }).sum::<f64>() / 3.0;

    println!("Summary:");
    println!("{}", "=".repeat(60));
    println!("{:<20} {:>15} {:>15} {:>15}", "", "TRAIN", "VALID", "TEST");
    println!("{:<20} {:>15.4} {:>15.4} {:>15.4}", "Hybrid baseline", hybrid_train, hybrid_valid, hybrid_test);
    println!("{:<20} {:>15.4} {:>15.4} {:>15.4}", "Evolved (layer-aware)", evolved_train, evolved_valid, evolved_test);

    let train_improvement = (hybrid_train - evolved_train) / hybrid_train * 100.0;
    let valid_improvement = (hybrid_valid - evolved_valid) / hybrid_valid * 100.0;
    let test_improvement = (hybrid_test - evolved_test) / hybrid_test * 100.0;

    println!();
    println!("Improvement (lower error = better):");
    println!("  TRAIN: {:.2}%", train_improvement);
    println!("  VALID: {:.2}%", valid_improvement);
    println!("  TEST:  {:.2}%", test_improvement);

    // JSON output
    println!("\nJSON:");
    println!("{{");
    println!("  \"evolved_train\": {:.6},", evolved_train);
    println!("  \"evolved_valid\": {:.6},", evolved_valid);
    println!("  \"evolved_test\": {:.6},", evolved_test);
    println!("  \"hybrid_train\": {:.6},", hybrid_train);
    println!("  \"hybrid_valid\": {:.6},", hybrid_valid);
    println!("  \"hybrid_test\": {:.6},", hybrid_test);
    println!("  \"improvement_train_pct\": {:.4},", train_improvement);
    println!("  \"improvement_valid_pct\": {:.4},", valid_improvement);
    println!("  \"improvement_test_pct\": {:.4}", test_improvement);
    println!("}}");
}
