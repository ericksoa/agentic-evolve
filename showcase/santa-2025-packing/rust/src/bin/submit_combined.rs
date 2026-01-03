//! Generate submission.csv using combined packer

use santa_packing::calculate_score;
use santa_packing::combined::CombinedPacker;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_file = args.get(1).map(|s| s.as_str()).unwrap_or("submission_combined.csv");

    println!("Generating submission using CombinedPacker...");

    let packer = CombinedPacker::default();
    let packings = packer.pack_all(200);

    // Validate
    let mut valid = true;
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("ERROR: n={} has {} trees (expected {})", i + 1, packing.trees.len(), i + 1);
            valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("ERROR: n={} has overlapping trees!", i + 1);
            valid = false;
        }
    }

    if !valid {
        eprintln!("Submission has errors!");
        std::process::exit(1);
    }

    let score = calculate_score(&packings);
    println!("Score: {:.4}", score);

    // Write CSV
    let file = File::create(output_file).expect("Failed to create file");
    let mut writer = BufWriter::new(file);

    writeln!(writer, "row_id,x,y,angle").unwrap();

    let mut row_id = 0;
    for packing in &packings {
        for tree in &packing.trees {
            writeln!(writer, "{},{},{},{}", row_id, tree.x, tree.y, tree.angle_deg).unwrap();
            row_id += 1;
        }
    }

    println!("Written to: {}", output_file);
}
