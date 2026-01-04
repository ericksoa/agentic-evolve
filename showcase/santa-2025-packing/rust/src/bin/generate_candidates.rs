//! Generate candidate packings for ML re-ranking
//!
//! Runs evolved multiple times and outputs all candidates in JSONL format
//! for Python-based ML selection.
//!
//! Usage: generate_candidates <max_n> <num_runs> [output_file]

use santa_packing::evolved::EvolvedPacker;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let output_file = args.get(3).map(|s| s.as_str()).unwrap_or("candidates.jsonl");

    eprintln!("Generating candidates: max_n={}, runs={}", max_n, num_runs);

    let file = File::create(output_file).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut total_candidates = 0;

    for run in 0..num_runs {
        if run % 5 == 0 || run == num_runs - 1 {
            eprintln!("  Run {}/{}", run + 1, num_runs);
        }

        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(max_n);

        for (n_idx, packing) in packings.iter().enumerate() {
            let n = n_idx + 1;

            // Skip if has overlaps
            if packing.has_overlaps() {
                continue;
            }

            let side_length = packing.side_length();

            // Format positions as [[x, y, rot], ...]
            let positions: Vec<[f64; 3]> = packing
                .trees
                .iter()
                .map(|tree| [tree.x, tree.y, tree.angle_deg])
                .collect();

            // Write as JSONL
            writeln!(
                writer,
                r#"{{"n": {}, "side_length": {}, "run": {}, "positions": {:?}}}"#,
                n, side_length, run, positions
            )
            .unwrap();

            total_candidates += 1;
        }
    }

    writer.flush().unwrap();
    eprintln!("Generated {} candidates to {}", total_candidates, output_file);
}
