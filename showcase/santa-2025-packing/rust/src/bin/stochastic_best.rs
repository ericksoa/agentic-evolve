//! Stochastic Best-of-N with parameter variation
//!
//! Each run uses slightly different parameters to explore
//! different parts of the solution space.

use santa_packing::calculate_score;
use santa_packing::evolved::{EvolvedConfig, EvolvedPacker};
use santa_packing::Packing;
use rand::Rng;
use std::time::Instant;

fn random_config(rng: &mut impl Rng, base: &EvolvedConfig) -> EvolvedConfig {
    // Vary parameters within reasonable ranges
    EvolvedConfig {
        search_attempts: rng.gen_range(150..250),
        direction_samples: rng.gen_range(48..80),
        sa_iterations: rng.gen_range(30000..50000),  // GEN109: increased range
        sa_initial_temp: rng.gen_range(0.35..0.55),
        sa_cooling_rate: rng.gen_range(0.99990..0.99996),
        sa_min_temp: base.sa_min_temp,
        translation_scale: rng.gen_range(0.045..0.065),
        rotation_granularity: base.rotation_granularity,
        center_pull_strength: rng.gen_range(0.06..0.10),
        sa_passes: rng.gen_range(1..4),
        early_exit_threshold: rng.gen_range(2000..3000),
        boundary_focus_prob: rng.gen_range(0.80..0.90),
        num_strategies: base.num_strategies,
        density_grid_resolution: base.density_grid_resolution,
        gap_penalty_weight: rng.gen_range(0.10..0.20),
        local_density_radius: base.local_density_radius,
        fill_move_prob: rng.gen_range(0.10..0.20),
        hot_restart_interval: rng.gen_range(600..1000),
        hot_restart_temp: rng.gen_range(0.30..0.40),
        elite_pool_size: rng.gen_range(2..5),
        compression_prob: rng.gen_range(0.15..0.25),
        wave_passes: rng.gen_range(4..7),
        late_stage_threshold: base.late_stage_threshold,
        fine_angle_step: base.fine_angle_step,
        swap_prob: base.swap_prob,
        // GEN109: new parameters
        combined_move_prob: rng.gen_range(0.15..0.25),
        squeeze_interval: rng.gen_range(4000..6000),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    eprintln!("Stochastic Best-of-{} with parameter variation (n=1..{})", num_runs, max_n);

    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let base_config = EvolvedConfig::default();

    // Collect all runs
    let mut all_packings: Vec<Vec<Packing>> = Vec::with_capacity(num_runs);

    for run in 0..num_runs {
        // Use default config for first run, random for rest
        let config = if run == 0 {
            EvolvedConfig::default()
        } else {
            random_config(&mut rng, &base_config)
        };

        if run % 5 == 0 || run == num_runs - 1 {
            eprintln!("  Run {}/{} (sa_iters={}, temp={:.3})",
                run + 1, num_runs, config.sa_iterations, config.sa_initial_temp);
        }

        let packer = EvolvedPacker { config };
        let packings = packer.pack_all(max_n);
        all_packings.push(packings);
    }

    // Select best for each n
    let mut best_packings: Vec<Packing> = Vec::with_capacity(max_n);
    let mut improvements = 0;

    for n_idx in 0..max_n {
        let mut best_side = f64::INFINITY;
        let mut best_packing: Option<&Packing> = None;
        let first_side = all_packings[0][n_idx].side_length();

        for run_packings in &all_packings {
            let side = run_packings[n_idx].side_length();
            if side < best_side && !run_packings[n_idx].has_overlaps() {
                best_side = side;
                best_packing = Some(&run_packings[n_idx]);
            }
        }

        let best = best_packing.unwrap_or(&all_packings[0][n_idx]);
        best_packings.push(best.clone());

        if best_side < first_side - 0.0001 {
            improvements += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let best_score = calculate_score(&best_packings);
    let first_score = calculate_score(&all_packings[0]);

    eprintln!("\nResults:");
    eprintln!("  First run (default):   {:.4}", first_score);
    eprintln!("  Stochastic best-of-{}: {:.4}", num_runs, best_score);
    eprintln!("  Improvement: {:.2}%", (first_score - best_score) / first_score * 100.0);
    eprintln!("  N improved: {}/{}", improvements, max_n);
    eprintln!("  Time: {:.1}s", elapsed);

    println!("[STOCHASTIC_BEST_SCORE={:.6}]", best_score);
}
