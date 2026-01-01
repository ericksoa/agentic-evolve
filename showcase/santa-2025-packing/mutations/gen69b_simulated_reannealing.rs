//! Evolved Packing Algorithm - Generation 69b SIMULATED REANNEALING
//!
//! MUTATION STRATEGY: Multiple complete SA runs with temperature reheating
//!
//! Hypothesis: Instead of hot restarts within a single SA run, do multiple
//! complete SA runs from scratch, keeping the best. This provides more
//! exploration than incremental restarts.
//!
//! Changes from Gen62:
//! - Run 3 independent SA optimizations per tree addition
//! - Each SA run starts with fresh temperature
//! - Keep best result across all runs
//! - Slightly fewer iterations per run to maintain time budget
//!
//! Target: Better exploration through multiple independent searches

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
    ConcentricRings,
}

pub struct EvolvedConfig {
    pub search_attempts: usize,
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,
    pub translation_scale: f64,
    pub center_pull_strength: f64,
    pub sa_runs: usize,  // NEW: Number of independent SA runs
    pub early_exit_threshold: usize,
    pub boundary_focus_prob: f64,
    pub compression_prob: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 200,
            sa_iterations: 15000,  // Reduced per-run (3 runs x 15k = 45k total vs 28k x 2 = 56k)
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.99993,
            sa_min_temp: 0.00001,
            translation_scale: 0.055,
            center_pull_strength: 0.07,
            sa_runs: 3,  // NEW: 3 independent runs
            early_exit_threshold: 2000,
            boundary_focus_prob: 0.85,
            compression_prob: 0.20,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left, Right, Top, Bottom, Corner, None,
}

pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        let strategies = [
            PlacementStrategy::ClockwiseSpiral,
            PlacementStrategy::CounterclockwiseSpiral,
            PlacementStrategy::Grid,
            PlacementStrategy::Random,
            PlacementStrategy::BoundaryFirst,
            PlacementStrategy::ConcentricRings,
        ];

        let mut strategy_trees: Vec<Vec<PlacedTree>> = vec![Vec::new(); strategies.len()];

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;

            for (s_idx, &strategy) in strategies.iter().enumerate() {
                let mut trees = strategy_trees[s_idx].clone();
                let new_tree = self.find_placement_with_strategy(&trees, n, strategy, &mut rng);
                trees.push(new_tree);

                // NEW: Multiple independent SA runs, keep best
                let mut run_best_trees = trees.clone();
                let mut run_best_side = compute_side_length(&trees);

                for _ in 0..self.config.sa_runs {
                    let mut run_trees = trees.clone();
                    self.local_search(&mut run_trees, n, &mut rng);

                    let run_side = compute_side_length(&run_trees);
                    if run_side < run_best_side {
                        run_best_side = run_side;
                        run_best_trees = run_trees;
                    }
                }

                strategy_trees[s_idx] = run_best_trees.clone();

                if run_best_side < best_side {
                    best_side = run_best_side;
                    best_trees = Some(run_best_trees);
                }
            }

            let best = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &best {
                packing.trees.push(t.clone());
            }
            packings.push(packing);

            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }

        packings
    }

    fn find_placement_with_strategy(
        &self,
        existing: &[PlacedTree],
        n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 45.0,
                PlacementStrategy::Random => rng.gen_range(0..8) as f64 * 45.0,
                PlacementStrategy::BoundaryFirst => 180.0,
                PlacementStrategy::ConcentricRings => 45.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        for attempt in 0..self.config.search_attempts {
            let dir = self.select_direction_for_strategy(n, current_width, current_height, strategy, attempt, rng);

            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, tree_angle);

                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = self.placement_score(&candidate, existing, n);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    fn select_direction_for_strategy(
        &self,
        n: usize,
        width: f64,
        height: f64,
        strategy: PlacementStrategy,
        attempt: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle) % (2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base + offset) % (2.0 * PI)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                let golden_angle = -PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle).rem_euclid(2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base - offset).rem_euclid(2.0 * PI)
            }
            PlacementStrategy::Grid => {
                let num_dirs = 16;
                let base_idx = attempt % num_dirs;
                let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                base + rng.gen_range(-0.03..0.03)
            }
            PlacementStrategy::Random => {
                let mix = rng.gen::<f64>();
                if mix < 0.5 {
                    rng.gen_range(0.0..2.0 * PI)
                } else if width < height {
                    let angle = if rng.gen() { 0.0 } else { PI };
                    angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                } else {
                    let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                    angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                }
            }
            PlacementStrategy::BoundaryFirst => {
                let prob = rng.gen::<f64>();
                if prob < 0.4 {
                    let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
                    corners[attempt % 4] + rng.gen_range(-0.1..0.1)
                } else if prob < 0.8 {
                    let edges = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
                    edges[attempt % 4] + rng.gen_range(-0.2..0.2)
                } else {
                    rng.gen_range(0.0..2.0 * PI)
                }
            }
            PlacementStrategy::ConcentricRings => {
                let ring = ((n as f64).sqrt() as usize).max(1);
                let trees_in_ring = (ring * 6).max(1);
                let position_in_ring = n % trees_in_ring;
                let base_angle = (position_in_ring as f64 / trees_in_ring as f64) * 2.0 * PI;
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 0.5 * PI;
                (base_angle + offset).rem_euclid(2.0 * PI)
            }
        }
    }

    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

        let mut pack_min_x = tree_min_x;
        let mut pack_min_y = tree_min_y;
        let mut pack_max_x = tree_max_x;
        let mut pack_max_y = tree_max_y;

        for t in existing {
            let (bx1, by1, bx2, by2) = t.bounds();
            pack_min_x = pack_min_x.min(bx1);
            pack_min_y = pack_min_y.min(by1);
            pack_max_x = pack_max_x.max(bx2);
            pack_max_y = pack_max_y.max(by2);
        }

        let width = pack_max_x - pack_min_x;
        let height = pack_max_y - pack_min_y;
        let side = width.max(height);

        let balance_penalty = (width - height).abs() * 0.10;

        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        side + balance_penalty + extension_penalty + center_penalty
    }

    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let mut temp = self.config.sa_initial_temp;
        let iterations = self.config.sa_iterations + n * 50;

        let mut iterations_without_improvement = 0;
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            let do_compression = rng.gen::<f64>() < self.config.compression_prob;

            if do_compression {
                let old_trees = trees.clone();
                let success = self.compression_move(trees, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        *trees = old_trees;
                        iterations_without_improvement += 1;
                    }
                } else {
                    *trees = old_trees;
                    iterations_without_improvement += 1;
                }
            } else {
                let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                    let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                    (bi.0, bi.1)
                } else {
                    (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                };

                let old_tree = trees[idx].clone();
                let success = self.sa_move(trees, idx, temp, edge, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        trees[idx] = old_tree;
                        iterations_without_improvement += 1;
                    }
                } else {
                    trees[idx] = old_tree;
                    iterations_without_improvement += 1;
                }
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    fn compression_move(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) -> bool {
        if trees.is_empty() {
            return false;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let idx = if rng.gen::<f64>() < 0.7 {
            let mut max_dist = 0.0;
            let mut max_idx = 0;
            for (i, tree) in trees.iter().enumerate() {
                let dx = tree.x - center_x;
                let dy = tree.y - center_y;
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    max_idx = i;
                }
            }
            max_idx
        } else {
            rng.gen_range(0..trees.len())
        };

        let old_x = trees[idx].x;
        let old_y = trees[idx].y;
        let old_angle = trees[idx].angle_deg;

        let dx = center_x - old_x;
        let dy = center_y - old_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.01 {
            return false;
        }

        let compression_factor = rng.gen_range(0.02..0.08);
        let new_x = old_x + dx * compression_factor;
        let new_y = old_y + dy * compression_factor;

        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

        !has_overlap(trees, idx)
    }

    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015;
        let mut boundary_info = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let edge = match (on_left, on_right, on_top, on_bottom) {
                (true, true, _, _) | (_, _, true, true) => BoundaryEdge::Corner,
                (true, _, true, _) | (true, _, _, true) => BoundaryEdge::Corner,
                (_, true, true, _) | (_, true, _, true) => BoundaryEdge::Corner,
                (true, false, false, false) => BoundaryEdge::Left,
                (false, true, false, false) => BoundaryEdge::Right,
                (false, false, true, false) => BoundaryEdge::Top,
                (false, false, false, true) => BoundaryEdge::Bottom,
                _ => continue,
            };
            boundary_info.push((i, edge));
        }
        boundary_info
    }

    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        let move_type = match edge {
            BoundaryEdge::Left => rng.gen_range(0..3),
            BoundaryEdge::Right => rng.gen_range(3..6),
            BoundaryEdge::Top => rng.gen_range(6..9),
            BoundaryEdge::Bottom => rng.gen_range(9..12),
            BoundaryEdge::Corner => rng.gen_range(0..4),
            BoundaryEdge::None => rng.gen_range(0..12),
        };

        match move_type % 4 {
            0 => {
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;
                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            2 => {
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
        }

        !has_overlap(trees, idx)
    }
}

fn is_valid(tree: &PlacedTree, existing: &[PlacedTree]) -> bool {
    existing.iter().all(|other| !tree.overlaps(other))
}

fn compute_side_length(trees: &[PlacedTree]) -> f64 {
    if trees.is_empty() { return 0.0; }
    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
    (max_x - min_x).max(max_y - min_y)
}

fn compute_bounds(trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }
    (min_x, min_y, max_x, max_y)
}

fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool {
    trees.iter().enumerate().any(|(i, tree)| i != idx && trees[idx].overlaps(tree))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_score;

    #[test]
    fn test_evolved_packer() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(20);
        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }

    #[test]
    fn test_evolved_score() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(50);
        let score = calculate_score(&packings);
        println!("Evolved score for n=1..50: {:.4}", score);
    }
}
