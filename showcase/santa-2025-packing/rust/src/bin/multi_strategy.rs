//! Multi-strategy optimizer
//!
//! Run different strategy configurations for each N,
//! pick the best result.

use santa_packing::calculate_score;
use santa_packing::evolved::{EvolvedConfig, EvolvedPacker};
use santa_packing::Packing;
use std::time::Instant;

/// Create configs with different strategy focuses
fn create_strategy_configs() -> Vec<(&'static str, EvolvedConfig)> {
    let base = EvolvedConfig::default();

    vec![
        ("default", EvolvedConfig::default()),

        ("high_sa", EvolvedConfig {
            sa_iterations: 50000,
            sa_initial_temp: 0.55,
            sa_cooling_rate: 0.99995,
            ..EvolvedConfig::default()
        }),

        ("low_temp", EvolvedConfig {
            sa_initial_temp: 0.30,
            sa_iterations: 35000,
            ..EvolvedConfig::default()
        }),

        ("more_waves", EvolvedConfig {
            wave_passes: 8,
            compression_prob: 0.30,
            ..EvolvedConfig::default()
        }),

        ("boundary_focus", EvolvedConfig {
            boundary_focus_prob: 0.95,
            search_attempts: 300,
            ..EvolvedConfig::default()
        }),

        ("center_pull", EvolvedConfig {
            center_pull_strength: 0.12,
            fill_move_prob: 0.25,
            ..EvolvedConfig::default()
        }),

        ("fast_cool", EvolvedConfig {
            sa_iterations: 40000,
            sa_cooling_rate: 0.99990,
            hot_restart_interval: 500,
            ..EvolvedConfig::default()
        }),

        ("slow_cool", EvolvedConfig {
            sa_iterations: 25000,
            sa_cooling_rate: 0.99997,
            hot_restart_interval: 1200,
            ..EvolvedConfig::default()
        }),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);

    let configs = create_strategy_configs();
    eprintln!("Multi-strategy optimization (n=1..{}, {} strategies)", max_n, configs.len());

    let start = Instant::now();

    // Run each strategy
    let mut all_packings: Vec<(&str, Vec<Packing>)> = Vec::new();

    for (name, config) in &configs {
        eprintln!("  Running strategy: {}", name);
        let packer = EvolvedPacker { config: config.clone() };
        let packings = packer.pack_all(max_n);
        all_packings.push((name, packings));
    }

    // Select best for each n
    let mut best_packings: Vec<Packing> = Vec::with_capacity(max_n);
    let mut strategy_wins: Vec<usize> = vec![0; configs.len()];

    for n_idx in 0..max_n {
        let mut best_side = f64::INFINITY;
        let mut best_idx = 0;
        let mut best_packing: Option<&Packing> = None;

        for (idx, (_, packings)) in all_packings.iter().enumerate() {
            let side = packings[n_idx].side_length();
            if side < best_side && !packings[n_idx].has_overlaps() {
                best_side = side;
                best_idx = idx;
                best_packing = Some(&packings[n_idx]);
            }
        }

        if let Some(p) = best_packing {
            best_packings.push(p.clone());
            strategy_wins[best_idx] += 1;
        } else {
            best_packings.push(all_packings[0].1[n_idx].clone());
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let best_score = calculate_score(&best_packings);
    let default_score = calculate_score(&all_packings[0].1);

    eprintln!("\nResults:");
    eprintln!("  Default only:    {:.4}", default_score);
    eprintln!("  Multi-strategy:  {:.4}", best_score);
    eprintln!("  Improvement: {:.2}%", (default_score - best_score) / default_score * 100.0);
    eprintln!("  Time: {:.1}s", elapsed);

    eprintln!("\nStrategy wins:");
    for (idx, (name, _)) in all_packings.iter().enumerate() {
        eprintln!("  {}: {}", name, strategy_wins[idx]);
    }

    println!("[MULTI_STRATEGY_SCORE={:.6}]", best_score);
}
