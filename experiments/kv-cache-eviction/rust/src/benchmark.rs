//! KV-Cache Eviction Benchmark
//!
//! Benchmarks various eviction scoring policies on synthetic attention patterns.
//! Measures attention reconstruction error at different compression ratios.

use std::time::Instant;

mod generate_data;

use generate_data::generate_benchmark_patterns;
use kv_cache::{
    baselines::{
        H2O, HybridBaseline, KnormPress, PositionCorrectedH2O, PyramidKV,
        RandomEviction, SnapKVLite, StreamingLLM, TOVA,
    },
    benchmark_scorer,
    evolved::Evolved,
    ScorerResult,
};

fn main() {
    println!("KV-Cache Eviction Policy Benchmark");
    println!("===================================\n");

    let start = Instant::now();

    // Generate benchmark data
    println!("Generating attention patterns...");
    let (train, valid, test) = generate_benchmark_patterns();
    println!(
        "Generated {} train, {} valid, {} test patterns\n",
        train.len(),
        valid.len(),
        test.len()
    );

    // Define all scorers to benchmark
    let scorers: Vec<Box<dyn kv_cache::EvictionScorer>> = vec![
        Box::new(StreamingLLM::new(4, 64)),
        Box::new(H2O::new(32)),
        Box::new(SnapKVLite::new(32)),
        Box::new(KnormPress),
        Box::new(TOVA::new(0.01)),
        Box::new(PyramidKV::new(48)),
        Box::new(HybridBaseline::new()),
        Box::new(PositionCorrectedH2O::new(32, 0.3)),
        Box::new(RandomEviction::new(42)),
        Box::new(Evolved),
    ];

    // Run benchmarks on all splits
    println!("Running benchmarks...\n");

    let mut train_results: Vec<ScorerResult> = Vec::new();
    let mut valid_results: Vec<ScorerResult> = Vec::new();
    let mut test_results: Vec<ScorerResult> = Vec::new();

    for scorer in &scorers {
        let train_result = benchmark_scorer(&**scorer, &train);
        let valid_result = benchmark_scorer(&**scorer, &valid);
        let test_result = benchmark_scorer(&**scorer, &test);

        train_results.push(train_result);
        valid_results.push(valid_result);
        test_results.push(test_result);
    }

    // Sort by TRAIN performance
    let mut indexed: Vec<(usize, &ScorerResult)> =
        train_results.iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        a.1.avg_error
            .partial_cmp(&b.1.avg_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print results
    println!("Results (sorted by TRAIN avg_error, lower is better):");
    println!("------------------------------------------------------\n");

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10}",
        "Scorer", "TRAIN", "VALID", "TEST", "Δ vs best"
    );
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10}",
        "", "avg_err", "avg_err", "avg_err", ""
    );
    println!("{}", "-".repeat(65));

    let best_train = indexed[0].1.avg_error;

    for (orig_idx, train_result) in &indexed {
        let valid_result = &valid_results[*orig_idx];
        let test_result = &test_results[*orig_idx];

        let delta = if train_result.avg_error > best_train {
            format!("+{:.2}%", (train_result.avg_error - best_train) / best_train * 100.0)
        } else {
            "baseline".to_string()
        };

        println!(
            "{:<25} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            train_result.name,
            train_result.avg_error,
            valid_result.avg_error,
            test_result.avg_error,
            delta
        );
    }

    println!("\n");

    // Detailed results for top 3
    println!("Detailed Results (Top 3):");
    println!("-------------------------\n");

    for (i, (orig_idx, _)) in indexed.iter().take(3).enumerate() {
        let train_result = &train_results[*orig_idx];
        let valid_result = &valid_results[*orig_idx];
        let test_result = &test_results[*orig_idx];

        println!("{}. {}", i + 1, train_result.name);
        println!(
            "   Compression 25%: TRAIN={:.4}, VALID={:.4}, TEST={:.4}",
            train_result.error_at_25, valid_result.error_at_25, test_result.error_at_25
        );
        println!(
            "   Compression 50%: TRAIN={:.4}, VALID={:.4}, TEST={:.4}",
            train_result.error_at_50, valid_result.error_at_50, test_result.error_at_50
        );
        println!(
            "   Compression 75%: TRAIN={:.4}, VALID={:.4}, TEST={:.4}",
            train_result.error_at_75, valid_result.error_at_75, test_result.error_at_75
        );
        println!();
    }

    // Output JSON for evolution tracking
    println!("JSON Output:");
    println!("------------");
    println!("{{");
    println!("  \"benchmark\": \"kv-cache-eviction\",");
    println!("  \"compression_ratios\": [0.25, 0.50, 0.75],");
    println!("  \"metric\": \"attention_reconstruction_error\",");
    println!("  \"lower_is_better\": true,");
    println!("  \"results\": [");

    for (i, (orig_idx, _)) in indexed.iter().enumerate() {
        let train_result = &train_results[*orig_idx];
        let valid_result = &valid_results[*orig_idx];
        let test_result = &test_results[*orig_idx];
        let comma = if i < indexed.len() - 1 { "," } else { "" };

        println!("    {{");
        println!("      \"name\": \"{}\",", train_result.name);
        println!("      \"train_avg_error\": {:.6},", train_result.avg_error);
        println!("      \"valid_avg_error\": {:.6},", valid_result.avg_error);
        println!("      \"test_avg_error\": {:.6},", test_result.avg_error);
        println!(
            "      \"train_by_compression\": [{:.6}, {:.6}, {:.6}],",
            train_result.error_at_25, train_result.error_at_50, train_result.error_at_75
        );
        println!(
            "      \"valid_by_compression\": [{:.6}, {:.6}, {:.6}],",
            valid_result.error_at_25, valid_result.error_at_50, valid_result.error_at_75
        );
        println!(
            "      \"test_by_compression\": [{:.6}, {:.6}, {:.6}]",
            test_result.error_at_25, test_result.error_at_50, test_result.error_at_75
        );
        println!("    }}{}", comma);
    }

    println!("  ]");
    println!("}}");

    let elapsed = start.elapsed();
    println!("\nBenchmark completed in {:.2}s", elapsed.as_secs_f64());
}
