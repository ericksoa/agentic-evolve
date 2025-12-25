use fibonacci::{Fibonacci, baselines::*, evolved::EvolvedFib};
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    ops_per_second: f64,
    correct: bool,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

fn verify_correctness<F: Fibonacci>(f: &F) -> bool {
    // Test known values
    let expected = [
        (0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5),
        (10, 55), (20, 6765), (30, 832040), (40, 102334155),
        (50, 12586269025), (60, 1548008755920), (70, 190392490709135),
        (80, 23416728348467685), (90, 2880067194370816120),
        (92, 7540113804746346429),
    ];

    for (n, exp) in expected {
        if f.fib(n) != exp {
            return false;
        }
    }
    true
}

fn benchmark<F: Fibonacci>(f: &F, test_values: &[u64], warmup_ms: u64, run_ms: u64) -> f64 {
    // Warmup
    let warmup_end = Instant::now() + Duration::from_millis(warmup_ms);
    while Instant::now() < warmup_end {
        for &n in test_values.iter().take(5) {
            std::hint::black_box(f.fib(n));
        }
    }

    // Benchmark
    let mut ops = 0u64;
    let start = Instant::now();
    let end = start + Duration::from_millis(run_ms);
    while Instant::now() < end {
        for &n in test_values {
            std::hint::black_box(f.fib(n));
            ops += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    ops as f64 / elapsed
}

fn main() {
    // Test values: mix of small, medium, and large
    // For naive recursion, we can only test small values
    let small_values: Vec<u64> = (0..=25).collect();
    let all_values: Vec<u64> = (0..=92).step_by(5).collect();

    let mut results = Vec::new();
    let mut all_correct = true;

    // Benchmark naive (only small values, or it takes forever)
    let naive = NaiveFib;
    let naive_correct = naive.fib(20) == 6765; // Quick sanity check
    all_correct &= naive_correct;
    let naive_ops = benchmark(&naive, &small_values[..15], 20, 200); // Only test 0-14
    results.push(BenchmarkResult {
        name: "naive".into(),
        ops_per_second: naive_ops,
        correct: naive_correct
    });

    // Benchmark iterative
    let iterative = IterativeFib;
    let correct = verify_correctness(&iterative);
    all_correct &= correct;
    let ops = benchmark(&iterative, &all_values, 50, 300);
    results.push(BenchmarkResult { name: "iterative".into(), ops_per_second: ops, correct });

    // Benchmark matrix
    let matrix = MatrixFib;
    let correct = verify_correctness(&matrix);
    all_correct &= correct;
    let ops = benchmark(&matrix, &all_values, 50, 300);
    results.push(BenchmarkResult { name: "matrix".into(), ops_per_second: ops, correct });

    // Benchmark lookup
    let lookup = LookupFib;
    let correct = verify_correctness(&lookup);
    all_correct &= correct;
    let ops = benchmark(&lookup, &all_values, 50, 300);
    results.push(BenchmarkResult { name: "lookup".into(), ops_per_second: ops, correct });

    // Benchmark evolved - test with appropriate values based on complexity
    let evolved = EvolvedFib;

    // Try to detect if evolved is still naive (very slow on n=25)
    let start = Instant::now();
    let test_result = evolved.fib(25);
    let elapsed = start.elapsed();

    let (evolved_correct, evolved_ops) = if elapsed.as_millis() > 500 {
        // Too slow - likely still naive recursion
        // Just verify small values and benchmark those
        let small_correct = evolved.fib(0) == 0 && evolved.fib(1) == 1 &&
                           evolved.fib(10) == 55 && evolved.fib(20) == 6765;
        let ops = benchmark(&evolved, &(0..=20).collect::<Vec<_>>(), 10, 100);
        (small_correct, ops)
    } else if elapsed.as_millis() > 50 {
        // Medium speed - can verify up to 40
        let medium_correct = evolved.fib(0) == 0 && evolved.fib(10) == 55 &&
                            evolved.fib(30) == 832040 && evolved.fib(40) == 102334155;
        let ops = benchmark(&evolved, &(0..=40).step_by(5).collect::<Vec<_>>(), 20, 200);
        (medium_correct, ops)
    } else {
        // Fast - verify all and benchmark full range
        let correct = verify_correctness(&evolved);
        let ops = benchmark(&evolved, &all_values, 50, 300);
        (correct, ops)
    };
    all_correct &= evolved_correct;

    results.push(BenchmarkResult {
        name: "evolved".into(),
        ops_per_second: evolved_ops,
        correct: evolved_correct
    });

    let full = FullResults { results, correctness: all_correct };
    println!("{}", serde_json::to_string(&full).unwrap());
}
