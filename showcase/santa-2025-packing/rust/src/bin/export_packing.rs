//! Export packing to JSON for Python hybrid pipeline
//!
//! Gen108: Rust-Python hybrid - export Rust's optimized packing
//! for further refinement in Python.

use santa_packing::evolved::EvolvedPacker;
use santa_packing::Packing;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn export_packing_json(packing: &Packing, output: &Path) -> std::io::Result<()> {
    let mut file = File::create(output)?;

    // Write JSON manually to avoid serde dependency
    writeln!(file, "{{")?;
    writeln!(file, "  \"n\": {},", packing.trees.len())?;
    writeln!(file, "  \"side_length\": {:.9},", packing.side_length())?;
    writeln!(file, "  \"trees\": [")?;

    for (i, tree) in packing.trees.iter().enumerate() {
        let comma = if i < packing.trees.len() - 1 { "," } else { "" };
        writeln!(
            file,
            "    {{\"x\": {:.9}, \"y\": {:.9}, \"angle_deg\": {:.9}}}{}",
            tree.x, tree.y, tree.angle_deg, comma
        )?;
    }

    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: export_packing <n> <output.json>");
        eprintln!("       export_packing <n> <output.json> <num_runs>");
        eprintln!("");
        eprintln!("Exports the best packing for n trees to JSON format.");
        eprintln!("Optional num_runs: run algorithm multiple times and pick best (default: 1)");
        std::process::exit(1);
    }

    let n: usize = args[1].parse().expect("n must be a positive integer");
    let output_path = Path::new(&args[2]);
    let num_runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    eprintln!("Generating packing for n={} ({} run{})", n, num_runs, if num_runs > 1 { "s" } else { "" });

    let mut best_packing: Option<Packing> = None;
    let mut best_side = f64::INFINITY;

    for run in 0..num_runs {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(n);
        let packing = &packings[n - 1];
        let side = packing.side_length();

        if side < best_side && !packing.has_overlaps() {
            best_side = side;
            best_packing = Some(packing.clone());
            eprintln!("  Run {}: side={:.6} (new best)", run + 1, side);
        } else if num_runs > 1 {
            eprintln!("  Run {}: side={:.6}", run + 1, side);
        }
    }

    let packing = best_packing.expect("No valid packing found");

    export_packing_json(&packing, output_path).expect("Failed to write JSON");

    eprintln!("Exported to {} (n={}, side={:.6})", output_path.display(), n, best_side);

    // Print compact output for scripting
    println!("{{\"n\": {}, \"side_length\": {:.9}}}", n, best_side);
}
