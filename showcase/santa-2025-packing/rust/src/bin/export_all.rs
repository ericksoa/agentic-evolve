//! Export all packings (n=1..max_n) to JSON for Python integration
//!
//! Gen109: Outputs JSON to stdout with best-of-N selection for each n

use santa_packing::evolved::EvolvedPacker;
use santa_packing::Packing;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    eprintln!("Generating packings n=1..{} with best-of-{}...", max_n, num_runs);

    let mut best_packings: Vec<Option<Packing>> = vec![None; max_n];
    let mut best_sides: Vec<f64> = vec![f64::INFINITY; max_n];

    for run in 0..num_runs {
        if run % 5 == 0 {
            eprintln!("  Run {}/{}", run + 1, num_runs);
        }

        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(max_n);

        for (n_idx, packing) in packings.iter().enumerate() {
            let side = packing.side_length();
            if side < best_sides[n_idx] && !packing.has_overlaps() {
                best_sides[n_idx] = side;
                best_packings[n_idx] = Some(packing.clone());
            }
        }
    }

    // Output JSON
    println!("{{");
    println!("  \"packings\": [");

    for n_idx in 0..max_n {
        let n = n_idx + 1;
        let packing = best_packings[n_idx].as_ref().unwrap();
        let comma = if n < max_n { "," } else { "" };

        println!("    {{");
        println!("      \"n\": {},", n);
        println!("      \"side\": {:.9},", best_sides[n_idx]);
        println!("      \"trees\": [");

        for (i, tree) in packing.trees.iter().enumerate() {
            let tree_comma = if i < packing.trees.len() - 1 { "," } else { "" };
            println!(
                "        {{\"x\": {:.9}, \"y\": {:.9}, \"angle\": {:.9}}}{}",
                tree.x, tree.y, tree.angle_deg, tree_comma
            );
        }

        println!("      ]");
        println!("    }}{}", comma);
    }

    println!("  ]");
    println!("}}");

    // Summary to stderr
    let score: f64 = best_sides.iter().enumerate()
        .map(|(i, &s)| s * s / (i + 1) as f64)
        .sum();

    eprintln!("\nTotal score: {:.4}", score);
    eprintln!("[EXPORT_ALL_SCORE={:.6}]", score);
}
