//! Benchmark the combined packer
//!
//! Measures score and validates correctness for competition submission.

use santa_packing::calculate_score;
use santa_packing::combined::CombinedPacker;
use santa_packing::evolved::EvolvedPacker;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);

    println!("Benchmarking CombinedPacker (n=1..{})", max_n);

    // First run evolved as baseline
    println!("\nRunning Evolved baseline...");
    let evolved_start = Instant::now();
    let evolved = EvolvedPacker::default();
    let evolved_packings = evolved.pack_all(max_n);
    let evolved_time = evolved_start.elapsed().as_secs_f64();
    let evolved_score = calculate_score(&evolved_packings);
    println!("  Evolved: score={:.4}, time={:.2}s", evolved_score, evolved_time);

    // Then run combined
    println!("\nRunning Combined...");
    let combined_start = Instant::now();
    let combined = CombinedPacker::default();
    let combined_packings = combined.pack_all(max_n);
    let combined_time = combined_start.elapsed().as_secs_f64();

    // Validate combined
    let mut valid = true;
    for (i, packing) in combined_packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("ERROR: n={} has {} trees (expected {})", i + 1, packing.trees.len(), i + 1);
            valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("ERROR: n={} has overlapping trees!", i + 1);
            valid = false;
        }
    }

    let combined_score = calculate_score(&combined_packings);

    // Count wins
    let mut combined_wins = 0;
    let mut evolved_wins = 0;
    let mut ties = 0;
    for i in 0..max_n {
        let c = combined_packings[i].side_length();
        let e = evolved_packings[i].side_length();
        if c < e - 0.0001 {
            combined_wins += 1;
        } else if e < c - 0.0001 {
            evolved_wins += 1;
        } else {
            ties += 1;
        }
    }

    println!("\nResults:");
    println!("  Combined: score={:.4}, time={:.2}s, valid={}", combined_score, combined_time, valid);
    println!("  Evolved:  score={:.4}, time={:.2}s", evolved_score, evolved_time);
    println!("  Improvement: {:.2}%", (evolved_score - combined_score) / evolved_score * 100.0);
    println!("  Wins: Combined={}, Evolved={}, Ties={}", combined_wins, evolved_wins, ties);

    // Output for submission
    println!("\n[COMBINED_SCORE={:.6}]", combined_score);
}
