//! Quick benchmark comparing Evolved vs Hybrid baseline only
//! Much faster than full benchmark by testing fewer scorers

mod generate_data;

use generate_data::generate_benchmark_patterns;
use kv_cache::{
    baselines::HybridBaseline,
    benchmark_scorer,
    evolved::Evolved,
};

fn main() {
    println!("Quick KV-Cache Benchmark: Evolved vs Hybrid");
    println!("=============================================\n");

    // Generate benchmark data
    println!("Generating attention patterns...");
    let (train, valid, test) = generate_benchmark_patterns();
    println!(
        "Generated {} train, {} valid, {} test patterns\n",
        train.len(),
        valid.len(),
        test.len()
    );

    println!("Running benchmarks...\n");

    // Test just the two key scorers
    let hybrid = HybridBaseline::new();
    let evolved = Evolved;

    let hybrid_train = benchmark_scorer(&hybrid, &train);
    let hybrid_valid = benchmark_scorer(&hybrid, &valid);
    let hybrid_test = benchmark_scorer(&hybrid, &test);

    let evolved_train = benchmark_scorer(&evolved, &train);
    let evolved_valid = benchmark_scorer(&evolved, &valid);
    let evolved_test = benchmark_scorer(&evolved, &test);

    println!("Results (lower is better):");
    println!("--------------------------\n");

    println!("{:<15} {:>10} {:>10} {:>10}", "Scorer", "TRAIN", "VALID", "TEST");
    println!("{}", "-".repeat(50));
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4}",
        "hybrid", hybrid_train.avg_error, hybrid_valid.avg_error, hybrid_test.avg_error
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4}",
        "evolved", evolved_train.avg_error, evolved_valid.avg_error, evolved_test.avg_error
    );

    // Calculate improvement
    let train_delta = (evolved_train.avg_error - hybrid_train.avg_error) / hybrid_train.avg_error * 100.0;
    let valid_delta = (evolved_valid.avg_error - hybrid_valid.avg_error) / hybrid_valid.avg_error * 100.0;
    let test_delta = (evolved_test.avg_error - hybrid_test.avg_error) / hybrid_test.avg_error * 100.0;

    println!("\nImprovement vs Hybrid Baseline:");
    println!("  TRAIN: {:.2}%", -train_delta);
    println!("  VALID: {:.2}%", -valid_delta);
    println!("  TEST:  {:.2}%", -test_delta);

    println!("\nDetailed Results:");
    println!("  Compression 25%: evolved={:.4}, hybrid={:.4}", evolved_train.error_at_25, hybrid_train.error_at_25);
    println!("  Compression 50%: evolved={:.4}, hybrid={:.4}", evolved_train.error_at_50, hybrid_train.error_at_50);
    println!("  Compression 75%: evolved={:.4}, hybrid={:.4}", evolved_train.error_at_75, hybrid_train.error_at_75);

    // JSON output for evolution tracking
    println!("\nJSON Output:");
    println!("{{");
    println!("  \"evolved_train\": {:.6},", evolved_train.avg_error);
    println!("  \"evolved_valid\": {:.6},", evolved_valid.avg_error);
    println!("  \"evolved_test\": {:.6},", evolved_test.avg_error);
    println!("  \"hybrid_train\": {:.6},", hybrid_train.avg_error);
    println!("  \"hybrid_valid\": {:.6},", hybrid_valid.avg_error);
    println!("  \"hybrid_test\": {:.6},", hybrid_test.avg_error);
    println!("  \"improvement_train_pct\": {:.4},", -train_delta);
    println!("  \"improvement_valid_pct\": {:.4},", -valid_delta);
    println!("  \"improvement_test_pct\": {:.4}", -test_delta);
    println!("}}");
}
