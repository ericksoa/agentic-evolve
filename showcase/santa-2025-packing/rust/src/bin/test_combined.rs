//! Test the combined strategy packer

use santa_packing::calculate_score;
use santa_packing::combined::CombinedPacker;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);

    println!("Testing CombinedPacker (n=1..{})", max_n);
    println!("Combines: Diamond/Hex init + Sparrow explore + Wave compact + Local search");
    println!("Plus: Evolved baseline with extra refinement");

    let packer = CombinedPacker::default();

    let start = Instant::now();
    let packings = packer.pack_all(max_n);
    let elapsed = start.elapsed().as_secs_f64();

    // Validate
    let mut valid = true;
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!(
                "ERROR: n={} has {} trees (expected {})",
                i + 1,
                packing.trees.len(),
                i + 1
            );
            valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("ERROR: n={} has overlapping trees!", i + 1);
            valid = false;
        }
    }

    let score = calculate_score(&packings);

    // Also run evolved for comparison
    println!("\nRunning Evolved for comparison...");
    let evolved_start = Instant::now();
    let evolved = santa_packing::evolved::EvolvedPacker::default();
    let evolved_packings = evolved.pack_all(max_n);
    let evolved_elapsed = evolved_start.elapsed().as_secs_f64();
    let evolved_score = calculate_score(&evolved_packings);

    println!("\nResults:");
    println!("  Combined score: {:.4} (time: {:.2}s)", score, elapsed);
    println!("  Evolved score:  {:.4} (time: {:.2}s)", evolved_score, evolved_elapsed);
    println!("  Improvement: {:.2}%", (evolved_score - score) / evolved_score * 100.0);
    println!("  Valid: {}", valid);

    // Print individual n scores
    println!("\nBreakdown (Combined | Evolved):");
    for (i, (comb, evol)) in packings.iter().zip(evolved_packings.iter()).enumerate() {
        let n = i + 1;
        let comb_side = comb.side_length();
        let evol_side = evol.side_length();
        let better = if comb_side < evol_side { "<" } else if comb_side > evol_side { ">" } else { "=" };
        println!(
            "  n={:3}: {:.4} {} {:.4}",
            n, comb_side, better, evol_side
        );
    }
}
