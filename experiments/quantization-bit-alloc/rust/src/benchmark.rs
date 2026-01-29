//! Benchmark for quantization bit allocation heuristics
//!
//! Evaluates heuristics on synthetic model architectures.
//! Metrics: compression ratio, accuracy retention, fitness score.

use quantization_bit_alloc::{
    baselines::*,
    evolved::Evolved,
    compute_fitness, evaluate_allocation,
    BitAllocationHeuristic, LayerInfo,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

/// Accuracy threshold for fitness calculation
const ACCURACY_THRESHOLD: f64 = 0.95;

#[derive(Serialize, Deserialize)]
struct Dataset {
    name: String,
    description: String,
    layers: Vec<LayerInfo>,
}

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    compression_ratio: f64,
    accuracy_retention: f64,
    avg_bits: f64,
    fitness: f64,
}

#[derive(Serialize)]
struct SplitResults {
    split: String,
    num_layers: usize,
    results: Vec<BenchmarkResult>,
}

#[derive(Serialize)]
struct FullBenchmark {
    benchmark: String,
    accuracy_threshold: f64,
    train: SplitResults,
    valid: SplitResults,
    test: SplitResults,
    evolved_metrics: EvolvedMetrics,
}

#[derive(Serialize, Clone, Copy)]
struct EvolvedMetrics {
    train_fitness: f64,
    valid_fitness: f64,
    test_fitness: f64,
    train_compression: f64,
    train_accuracy: f64,
    train_avg_bits: f64,
}

fn load_dataset(path: &str) -> Dataset {
    let contents = fs::read_to_string(path)
        .expect(&format!("Failed to read {}", path));
    serde_json::from_str(&contents)
        .expect(&format!("Failed to parse {}", path))
}

fn evaluate_heuristic<H: BitAllocationHeuristic>(
    name: &str,
    heuristic: &H,
    layers: &[LayerInfo],
) -> BenchmarkResult {
    let result = evaluate_allocation(heuristic, layers);
    let fitness = compute_fitness(&result, ACCURACY_THRESHOLD);

    BenchmarkResult {
        name: name.to_string(),
        compression_ratio: result.compression_ratio,
        accuracy_retention: result.accuracy_retention,
        avg_bits: result.avg_bits,
        fitness,
    }
}

fn evaluate_all(layers: &[LayerInfo]) -> Vec<BenchmarkResult> {
    vec![
        evaluate_heuristic("uniform_int8", &UniformInt8, layers),
        evaluate_heuristic("uniform_int4", &UniformInt4, layers),
        evaluate_heuristic("first_last_fp16", &FirstLastFP16, layers),
        evaluate_heuristic("sensitivity_based", &SensitivityBased, layers),
        evaluate_heuristic("layer_type_aware", &LayerTypeAware, layers),
        evaluate_heuristic("position_aware", &PositionAware, layers),
        evaluate_heuristic("hybrid_baseline", &HybridBaseline, layers),
        evaluate_heuristic("greedy_sensitivity", &GreedySensitivity::default(), layers),
        evaluate_heuristic("pareto_optimal", &ParetoOptimal::default(), layers),
        evaluate_heuristic("EVOLVED", &Evolved, layers),
    ]
}

fn main() {
    // Load datasets
    let train = load_dataset("data/train.json");
    let valid = load_dataset("data/valid.json");
    let test = load_dataset("data/test.json");

    // Evaluate all heuristics
    let start = Instant::now();

    let train_results = evaluate_all(&train.layers);
    let valid_results = evaluate_all(&valid.layers);
    let test_results = evaluate_all(&test.layers);

    let elapsed = start.elapsed();

    // Extract evolved metrics
    let evolved_train = train_results.iter().find(|r| r.name == "EVOLVED").unwrap();
    let evolved_valid = valid_results.iter().find(|r| r.name == "EVOLVED").unwrap();
    let evolved_test = test_results.iter().find(|r| r.name == "EVOLVED").unwrap();

    let evolved_metrics = EvolvedMetrics {
        train_fitness: evolved_train.fitness,
        valid_fitness: evolved_valid.fitness,
        test_fitness: evolved_test.fitness,
        train_compression: evolved_train.compression_ratio,
        train_accuracy: evolved_train.accuracy_retention,
        train_avg_bits: evolved_train.avg_bits,
    };

    // Build full benchmark output
    let benchmark = FullBenchmark {
        benchmark: "quantization-bit-alloc".to_string(),
        accuracy_threshold: ACCURACY_THRESHOLD,
        train: SplitResults {
            split: "TRAIN".to_string(),
            num_layers: train.layers.len(),
            results: train_results,
        },
        valid: SplitResults {
            split: "VALID".to_string(),
            num_layers: valid.layers.len(),
            results: valid_results,
        },
        test: SplitResults {
            split: "TEST".to_string(),
            num_layers: test.layers.len(),
            results: test_results,
        },
        evolved_metrics,
    };

    // Print JSON output
    println!("{}", serde_json::to_string_pretty(&benchmark).unwrap());

    // Print human-readable summary to stderr
    eprintln!("\n=== Quantization Bit Allocation Benchmark ===");
    eprintln!("Accuracy threshold: {:.1}%", ACCURACY_THRESHOLD * 100.0);
    eprintln!("Evaluation time: {:?}\n", elapsed);

    for (name, results) in [("TRAIN", &benchmark.train), ("VALID", &benchmark.valid), ("TEST", &benchmark.test)] {
        eprintln!("=== {} ({} layers) ===", name, results.num_layers);
        eprintln!("{:<25} {:>8} {:>8} {:>8} {:>10}",
            "Heuristic", "Compress", "Accuracy", "AvgBits", "Fitness");
        eprintln!("{}", "-".repeat(65));

        let mut sorted_results: Vec<_> = results.results.iter().collect();
        sorted_results.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        for r in sorted_results {
            let marker = if r.name == "EVOLVED" { " **" } else { "" };
            eprintln!("{:<25} {:>7.2}x {:>7.1}% {:>8.2} {:>10.4}{}",
                r.name,
                r.compression_ratio,
                r.accuracy_retention * 100.0,
                r.avg_bits,
                r.fitness,
                marker);
        }
        eprintln!();
    }

    // Print best baseline for comparison
    let best_baseline = benchmark.train.results.iter()
        .filter(|r| r.name != "EVOLVED")
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();

    eprintln!("=== Summary ===");
    eprintln!("Best baseline: {} (fitness: {:.4})", best_baseline.name, best_baseline.fitness);
    eprintln!("Evolved:       EVOLVED (fitness: {:.4})", evolved_metrics.train_fitness);

    let improvement = (evolved_metrics.train_fitness / best_baseline.fitness - 1.0) * 100.0;
    if improvement > 0.0 {
        eprintln!("Improvement:   +{:.2}%", improvement);
    } else {
        eprintln!("Improvement:   {:.2}% (needs work)", improvement);
    }
}
