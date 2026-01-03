//! Global Optimization for Tree Packing
//!
//! Unlike the greedy incremental approach, this optimizes all tree positions
//! simultaneously using particle swarm optimization (PSO) or differential
//! evolution-like approaches.

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Tree placement as optimization variables
#[derive(Clone, Debug)]
struct TreePlacement {
    x: f64,
    y: f64,
    rotation_idx: usize, // 0-7 for 45 degree increments
}

impl TreePlacement {
    fn to_placed_tree(&self) -> PlacedTree {
        PlacedTree::new(self.x, self.y, self.rotation_idx as f64 * 45.0)
    }
}

/// Compute bounding box side length for a set of placements
fn compute_side(placements: &[TreePlacement]) -> f64 {
    if placements.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for p in placements {
        let tree = p.to_placed_tree();
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

/// Count number of overlapping pairs
fn count_overlaps(placements: &[TreePlacement]) -> usize {
    let trees: Vec<PlacedTree> = placements.iter().map(|p| p.to_placed_tree()).collect();
    let mut count = 0;

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if trees[i].overlaps(&trees[j]) {
                count += 1;
            }
        }
    }

    count
}

/// Objective function: side length + large penalty for overlaps
fn objective(placements: &[TreePlacement]) -> f64 {
    let side = compute_side(placements);
    let overlaps = count_overlaps(placements);

    // Large penalty for overlaps
    side + overlaps as f64 * 10.0
}

/// Check if placements are valid (no overlaps)
fn is_valid(placements: &[TreePlacement]) -> bool {
    count_overlaps(placements) == 0
}

/// Initialize placements using spiral pattern
fn initialize_spiral(n: usize, rng: &mut impl Rng) -> Vec<TreePlacement> {
    let mut placements = Vec::with_capacity(n);
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());

    for i in 0..n {
        let angle = i as f64 * golden_angle;
        let dist = 0.5 + (i as f64 * 0.4).sqrt();

        let x = dist * angle.cos();
        let y = dist * angle.sin();
        let rotation_idx = rng.gen_range(0..8);

        placements.push(TreePlacement { x, y, rotation_idx });
    }

    placements
}

/// Initialize placements using grid pattern
fn initialize_grid(n: usize, rng: &mut impl Rng) -> Vec<TreePlacement> {
    let mut placements = Vec::with_capacity(n);

    // Estimate grid size needed
    let side_count = ((n as f64).sqrt().ceil() as usize).max(1);
    let spacing = 0.8; // Tree spacing

    let mut idx = 0;
    'outer: for row in 0..side_count {
        for col in 0..side_count {
            if idx >= n {
                break 'outer;
            }

            let x = (col as f64 - side_count as f64 / 2.0) * spacing;
            let y = (row as f64 - side_count as f64 / 2.0) * spacing;
            let rotation_idx = rng.gen_range(0..8);

            placements.push(TreePlacement { x, y, rotation_idx });
            idx += 1;
        }
    }

    placements
}

/// Differential evolution-style mutation
fn mutate(placements: &[TreePlacement], scale: f64, rng: &mut impl Rng) -> Vec<TreePlacement> {
    let mut result = placements.to_vec();

    // Pick random tree to modify
    let idx = rng.gen_range(0..result.len());

    // 70% chance position change, 30% chance rotation change
    if rng.gen::<f64>() < 0.7 {
        result[idx].x += rng.gen_range(-scale..scale);
        result[idx].y += rng.gen_range(-scale..scale);
    } else {
        result[idx].rotation_idx = rng.gen_range(0..8);
    }

    result
}

/// Crossover between two solutions
fn crossover(
    p1: &[TreePlacement],
    p2: &[TreePlacement],
    rng: &mut impl Rng,
) -> Vec<TreePlacement> {
    let n = p1.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen::<f64>() < 0.5 {
            result.push(p1[i].clone());
        } else {
            result.push(p2[i].clone());
        }
    }

    result
}

/// Local search to resolve overlaps
fn repair_overlaps(placements: &mut [TreePlacement], max_iter: usize, rng: &mut impl Rng) {
    for _ in 0..max_iter {
        if is_valid(placements) {
            break;
        }

        let trees: Vec<PlacedTree> = placements.iter().map(|p| p.to_placed_tree()).collect();

        // Find overlapping pairs
        for i in 0..trees.len() {
            for j in (i + 1)..trees.len() {
                if trees[i].overlaps(&trees[j]) {
                    // Push trees apart
                    let dx = placements[j].x - placements[i].x;
                    let dy = placements[j].y - placements[i].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    let push = 0.05;
                    placements[i].x -= push * dx / dist;
                    placements[i].y -= push * dy / dist;
                    placements[j].x += push * dx / dist;
                    placements[j].y += push * dy / dist;

                    // Sometimes try rotation
                    if rng.gen::<f64>() < 0.1 {
                        placements[i].rotation_idx = rng.gen_range(0..8);
                        placements[j].rotation_idx = rng.gen_range(0..8);
                    }
                }
            }
        }
    }
}

/// Compact placements toward center
fn compact(placements: &mut [TreePlacement], iterations: usize) {
    for _ in 0..iterations {
        if placements.is_empty() {
            return;
        }

        // Find center
        let cx: f64 = placements.iter().map(|p| p.x).sum::<f64>() / placements.len() as f64;
        let cy: f64 = placements.iter().map(|p| p.y).sum::<f64>() / placements.len() as f64;

        for i in 0..placements.len() {
            let dx = cx - placements[i].x;
            let dy = cy - placements[i].y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.01 {
                let old_x = placements[i].x;
                let old_y = placements[i].y;

                // Try moving slightly toward center
                placements[i].x += dx * 0.01;
                placements[i].y += dy * 0.01;

                // Check if this creates overlap
                let trees: Vec<PlacedTree> =
                    placements.iter().map(|pl| pl.to_placed_tree()).collect();

                let my_tree = placements[i].to_placed_tree();
                let mut has_overlap = false;
                for (j, other) in trees.iter().enumerate() {
                    if i != j && my_tree.overlaps(other) {
                        has_overlap = true;
                        break;
                    }
                }

                if has_overlap {
                    placements[i].x = old_x;
                    placements[i].y = old_y;
                }
            }
        }
    }
}

/// Global optimization using differential evolution
pub fn optimize_global(n: usize, iterations: usize, population_size: usize) -> Vec<TreePlacement> {
    let mut rng = rand::thread_rng();

    // Initialize population
    let mut population: Vec<Vec<TreePlacement>> = (0..population_size)
        .map(|i| {
            if i % 2 == 0 {
                initialize_spiral(n, &mut rng)
            } else {
                initialize_grid(n, &mut rng)
            }
        })
        .collect();

    // Repair initial population
    for p in population.iter_mut() {
        repair_overlaps(p, 1000, &mut rng);
    }

    // Track best solution
    let mut best = population[0].clone();
    let mut best_score = if is_valid(&best) {
        compute_side(&best)
    } else {
        f64::INFINITY
    };

    for iter in 0..iterations {
        let temp = 0.3 * (1.0 - iter as f64 / iterations as f64);
        let mutation_scale = 0.1 * (1.0 + temp);

        // Evaluate population
        let mut scored: Vec<(f64, usize)> = population
            .iter()
            .enumerate()
            .map(|(i, p)| (objective(p), i))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Update best
        let (best_obj, best_idx) = scored[0];
        if is_valid(&population[best_idx]) {
            let side = compute_side(&population[best_idx]);
            if side < best_score {
                best_score = side;
                best = population[best_idx].clone();
            }
        }

        // Selection: keep top half
        let survivors: Vec<Vec<TreePlacement>> = scored
            .iter()
            .take(population_size / 2)
            .map(|(_, i)| population[*i].clone())
            .collect();

        // Generate new population
        let mut new_pop = survivors.clone();

        while new_pop.len() < population_size {
            let p1 = &survivors[rng.gen_range(0..survivors.len())];
            let p2 = &survivors[rng.gen_range(0..survivors.len())];

            let mut child = if rng.gen::<f64>() < 0.5 {
                crossover(p1, p2, &mut rng)
            } else {
                mutate(p1, mutation_scale, &mut rng)
            };

            repair_overlaps(&mut child, 200, &mut rng);
            new_pop.push(child);
        }

        population = new_pop;

        // Periodic compaction
        if iter % 50 == 0 {
            for p in population.iter_mut() {
                compact(p, 20);
            }
        }
    }

    // Final compaction of best
    compact(&mut best, 100);
    repair_overlaps(&mut best, 500, &mut rng);

    best
}

/// Convert optimized placements to Packing
pub fn placements_to_packing(placements: &[TreePlacement]) -> Packing {
    let mut packing = Packing::new();
    for p in placements {
        packing.trees.push(p.to_placed_tree());
    }
    packing
}

pub struct GlobalOptPacker {
    pub iterations: usize,
    pub population_size: usize,
}

impl Default for GlobalOptPacker {
    fn default() -> Self {
        Self {
            iterations: 500,
            population_size: 20,
        }
    }
}

impl GlobalOptPacker {
    pub fn pack(&self, n: usize) -> Packing {
        let placements = optimize_global(n, self.iterations, self.population_size);
        placements_to_packing(&placements)
    }

    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        (1..=max_n).map(|n| self.pack(n)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_opt_small() {
        let placements = optimize_global(5, 100, 10);
        assert_eq!(placements.len(), 5);
        assert!(is_valid(&placements));
    }
}
