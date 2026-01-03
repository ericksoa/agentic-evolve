//! Combined strategies - best ideas from all approaches
//!
//! Key combinations:
//! 1. Multi-strategy selector: Run all, pick best per N
//! 2. Sparrow exploration → Evolved refinement
//! 3. Pattern seeding → SA optimization
//! 4. Evolved base → Intensive compression

use crate::{Packing, PlacedTree};
use crate::evolved::EvolvedPacker;
use rand::Rng;
use std::f64::consts::PI;

/// Tree state for manipulation
#[derive(Clone, Copy, Debug)]
struct Tree {
    x: f64,
    y: f64,
    rot: usize, // 0-7
}

impl Tree {
    fn to_placed(&self) -> PlacedTree {
        PlacedTree::new(self.x, self.y, self.rot as f64 * 45.0)
    }

    fn from_placed(p: &PlacedTree) -> Self {
        Self {
            x: p.x,
            y: p.y,
            rot: ((p.angle_deg / 45.0).round() as usize) % 8,
        }
    }
}

fn bbox_side(trees: &[Tree]) -> f64 {
    if trees.is_empty() { return 0.0; }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        let p = t.to_placed();
        let (bx1, by1, bx2, by2) = p.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

fn is_valid(trees: &[Tree]) -> bool {
    let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed()).collect();
    for i in 0..placed.len() {
        for j in (i+1)..placed.len() {
            if placed[i].overlaps(&placed[j]) {
                return false;
            }
        }
    }
    true
}

fn penetration_depth(t1: &PlacedTree, t2: &PlacedTree) -> f64 {
    let (ax1, ay1, ax2, ay2) = t1.bounds();
    let (bx1, by1, bx2, by2) = t2.bounds();

    let overlap_x = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let overlap_y = (ay2.min(by2) - ay1.max(by1)).max(0.0);

    if overlap_x == 0.0 || overlap_y == 0.0 { return 0.0; }
    if !t1.overlaps(t2) { return 0.0; }

    let c1x = (ax1 + ax2) / 2.0;
    let c1y = (ay1 + ay2) / 2.0;
    let c2x = (bx1 + bx2) / 2.0;
    let c2y = (by1 + by2) / 2.0;

    let dist = ((c2x - c1x).powi(2) + (c2y - c1y).powi(2)).sqrt();
    let min_sep = ((ax2 - ax1 + bx2 - bx1).powi(2) + (ay2 - ay1 + by2 - by1).powi(2)).sqrt() * 0.35;

    (min_sep - dist).max(0.01)
}

fn center_trees(trees: &mut [Tree]) {
    if trees.is_empty() { return; }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees.iter() {
        let p = t.to_placed();
        let (bx1, by1, bx2, by2) = p.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;

    for t in trees.iter_mut() {
        t.x -= cx;
        t.y -= cy;
    }
}

// ============================================
// Strategy 1: Diamond pattern initialization
// ============================================

fn diamond_init(n: usize) -> Vec<Tree> {
    let mut trees = Vec::with_capacity(n);

    // Staggered grid optimized for 45° rotated trees
    let spacing_x = 0.75;
    let spacing_y = 0.70;
    let offset = 0.375;

    let cols = ((n as f64).sqrt().ceil() as usize).max(1);

    let mut count = 0;
    let mut row = 0;
    while count < n {
        let row_offset = if row % 2 == 0 { 0.0 } else { offset };
        for col in 0..cols {
            if count >= n { break; }
            let x = (col as f64 - cols as f64 / 2.0) * spacing_x + row_offset;
            let y = (row as f64 - (n / cols) as f64 / 2.0) * spacing_y;
            // Alternate rotations for interlocking
            let rot = if (row + col) % 2 == 0 { 1 } else { 5 }; // 45° and 225°
            trees.push(Tree { x, y, rot });
            count += 1;
        }
        row += 1;
    }

    trees
}

// ============================================
// Strategy 2: Hexagonal pattern initialization
// ============================================

fn hexagonal_init(n: usize) -> Vec<Tree> {
    let mut trees = Vec::with_capacity(n);

    let spacing = 0.80;
    let row_height = spacing * 0.866; // sqrt(3)/2

    let cols = ((n as f64).sqrt().ceil() as usize).max(1);

    let mut count = 0;
    let mut row = 0;
    while count < n {
        let row_offset = if row % 2 == 0 { 0.0 } else { spacing / 2.0 };
        for col in 0..cols + 1 {
            if count >= n { break; }
            let x = (col as f64 - cols as f64 / 2.0) * spacing + row_offset;
            let y = (row as f64 - (n / cols) as f64 / 2.0) * row_height;
            let rot = ((row + col) % 8) as usize;
            trees.push(Tree { x, y, rot });
            count += 1;
        }
        row += 1;
    }

    trees
}

// ============================================
// Strategy 3: Sparrow-style exploration
// ============================================

fn sparrow_explore(n: usize, init: Vec<Tree>, iterations: usize) -> Vec<Tree> {
    let mut rng = rand::thread_rng();
    let mut trees = init;
    let mut weights: Vec<Vec<f64>> = vec![vec![1.0; n]; n];

    let mut best_trees: Option<Vec<Tree>> = None;
    let mut best_side = f64::INFINITY;

    for iter in 0..iterations {
        let progress = iter as f64 / iterations as f64;
        let step = 0.05 * (1.0 - 0.5 * progress);

        // Find and resolve overlaps
        let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed()).collect();
        let mut any_overlap = false;

        for i in 0..n {
            for j in (i+1)..n {
                let depth = penetration_depth(&placed[i], &placed[j]);
                if depth > 0.0 {
                    any_overlap = true;
                    weights[i][j] = (weights[i][j] * 1.05).min(50.0);
                    weights[j][i] = weights[i][j];

                    // Push apart
                    let dx = trees[j].x - trees[i].x;
                    let dy = trees[j].y - trees[i].y;
                    let dist = (dx*dx + dy*dy).sqrt().max(0.01);
                    let push = step * weights[i][j].sqrt();

                    trees[i].x -= push * dx / dist;
                    trees[i].y -= push * dy / dist;
                    trees[j].x += push * dx / dist;
                    trees[j].y += push * dy / dist;

                    if rng.gen::<f64>() < 0.1 {
                        trees[i].rot = rng.gen_range(0..8);
                    }
                } else {
                    weights[i][j] *= 0.98;
                    weights[j][i] = weights[i][j];
                }
            }
        }

        if !any_overlap {
            center_trees(&mut trees);
            let side = bbox_side(&trees);
            if side < best_side {
                best_side = side;
                best_trees = Some(trees.clone());
            }

            // Try compression
            let shrink = 0.01 * (1.0 - progress);
            for t in trees.iter_mut() {
                t.x *= 1.0 - shrink;
                t.y *= 1.0 - shrink;
            }
        }

        if iter % 100 == 0 {
            center_trees(&mut trees);
        }
    }

    best_trees.unwrap_or(trees)
}

// ============================================
// Strategy 4: Wave compaction (from evolved)
// ============================================

fn wave_compaction(trees: &mut [Tree], passes: usize) {
    for pass in 0..passes {
        if trees.is_empty() { return; }

        let (min_x, min_y, max_x, max_y) = {
            let mut mi_x = f64::INFINITY;
            let mut mi_y = f64::INFINITY;
            let mut ma_x = f64::NEG_INFINITY;
            let mut ma_y = f64::NEG_INFINITY;
            for t in trees.iter() {
                let p = t.to_placed();
                let (bx1, by1, bx2, by2) = p.bounds();
                mi_x = mi_x.min(bx1);
                mi_y = mi_y.min(by1);
                ma_x = ma_x.max(bx2);
                ma_y = ma_y.max(by2);
            }
            (mi_x, mi_y, ma_x, ma_y)
        };

        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;

        // Sort by distance from center (outside-in for first 4, inside-out for last)
        let mut indices: Vec<(usize, f64)> = trees.iter().enumerate()
            .map(|(i, t)| (i, ((t.x - cx).powi(2) + (t.y - cy).powi(2)).sqrt()))
            .collect();

        if pass < 4 {
            indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else {
            indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        // Move each tree toward center
        for (idx, _) in indices {
            let old = trees[idx];
            let dx = cx - old.x;
            let dy = cy - old.y;
            let dist = (dx*dx + dy*dy).sqrt();

            if dist < 0.02 { continue; }

            for step in [0.10, 0.05, 0.02, 0.01, 0.005] {
                trees[idx].x = old.x + dx * step;
                trees[idx].y = old.y + dy * step;

                if !is_valid(trees) {
                    trees[idx] = old;
                } else {
                    break;
                }
            }
        }
    }
}

// ============================================
// Strategy 5: Intensive local search
// ============================================

fn local_search(trees: &mut [Tree], iterations: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let n = trees.len();
    if n == 0 { return 0.0; }

    let mut best_side = bbox_side(trees);
    let mut best_trees = trees.to_vec();

    for iter in 0..iterations {
        let progress = iter as f64 / iterations as f64;
        let step = 0.03 * (1.0 - 0.7 * progress);

        let idx = rng.gen_range(0..n);
        let old = trees[idx];

        match rng.gen_range(0..5) {
            0 => {
                trees[idx].x += rng.gen_range(-step..step);
                trees[idx].y += rng.gen_range(-step..step);
            }
            1 => {
                // Move toward center
                let (min_x, min_y, max_x, max_y) = {
                    let mut mi_x = f64::INFINITY;
                    let mut mi_y = f64::INFINITY;
                    let mut ma_x = f64::NEG_INFINITY;
                    let mut ma_y = f64::NEG_INFINITY;
                    for t in trees.iter() {
                        let p = t.to_placed();
                        let (bx1, by1, bx2, by2) = p.bounds();
                        mi_x = mi_x.min(bx1);
                        mi_y = mi_y.min(by1);
                        ma_x = ma_x.max(bx2);
                        ma_y = ma_y.max(by2);
                    }
                    (mi_x, mi_y, ma_x, ma_y)
                };
                let cx = (min_x + max_x) / 2.0;
                let cy = (min_y + max_y) / 2.0;
                trees[idx].x += (cx - old.x) * step * 2.0;
                trees[idx].y += (cy - old.y) * step * 2.0;
            }
            2 => {
                trees[idx].rot = rng.gen_range(0..8);
            }
            3 => {
                // Move + rotate
                trees[idx].x += rng.gen_range(-step..step);
                trees[idx].y += rng.gen_range(-step..step);
                if rng.gen::<f64>() < 0.3 {
                    trees[idx].rot = rng.gen_range(0..8);
                }
            }
            _ => {
                // Try all 8 rotations at current position
                let mut best_rot = old.rot;
                let mut best_local_side = f64::INFINITY;
                for r in 0..8 {
                    trees[idx].rot = r;
                    if is_valid(trees) {
                        let side = bbox_side(trees);
                        if side < best_local_side {
                            best_local_side = side;
                            best_rot = r;
                        }
                    }
                }
                trees[idx].rot = best_rot;
            }
        }

        if is_valid(trees) {
            let side = bbox_side(trees);
            if side < best_side {
                best_side = side;
                best_trees.copy_from_slice(trees);
            } else if side > best_side * 1.002 {
                trees[idx] = old;
            }
        } else {
            trees[idx] = old;
        }
    }

    trees.copy_from_slice(&best_trees);
    best_side
}

// ============================================
// Combined strategy packer
// ============================================

pub struct CombinedPacker {
    pub sparrow_iters: usize,
    pub local_iters: usize,
    pub wave_passes: usize,
}

impl Default for CombinedPacker {
    fn default() -> Self {
        Self {
            sparrow_iters: 3000,
            local_iters: 10000,
            wave_passes: 5,
        }
    }
}

impl CombinedPacker {
    pub fn pack(&self, n: usize) -> Packing {
        if n == 0 { return Packing::new(); }
        if n == 1 {
            let mut p = Packing::new();
            p.trees.push(PlacedTree::new(0.0, 0.0, 45.0));
            return p;
        }

        let mut best_packing: Option<Packing> = None;
        let mut best_side = f64::INFINITY;

        // Strategy A: Diamond init → Sparrow explore → Wave compact → Local search
        {
            let init = diamond_init(n);
            let mut trees = sparrow_explore(n, init, self.sparrow_iters);
            if is_valid(&trees) {
                wave_compaction(&mut trees, self.wave_passes);
                local_search(&mut trees, self.local_iters);
                center_trees(&mut trees);

                if is_valid(&trees) {
                    let side = bbox_side(&trees);
                    if side < best_side {
                        best_side = side;
                        let mut p = Packing::new();
                        for t in &trees {
                            p.trees.push(t.to_placed());
                        }
                        best_packing = Some(p);
                    }
                }
            }
        }

        // Strategy B: Hexagonal init → Sparrow explore → Wave compact → Local search
        {
            let init = hexagonal_init(n);
            let mut trees = sparrow_explore(n, init, self.sparrow_iters);
            if is_valid(&trees) {
                wave_compaction(&mut trees, self.wave_passes);
                local_search(&mut trees, self.local_iters);
                center_trees(&mut trees);

                if is_valid(&trees) {
                    let side = bbox_side(&trees);
                    if side < best_side {
                        best_side = side;
                        let mut p = Packing::new();
                        for t in &trees {
                            p.trees.push(t.to_placed());
                        }
                        best_packing = Some(p);
                    }
                }
            }
        }

        // Strategy C: Evolved base → Extra local refinement
        {
            let evolved = EvolvedPacker::default();
            let evolved_packings = evolved.pack_all(n);
            let evolved_packing = &evolved_packings[n - 1];

            let mut trees: Vec<Tree> = evolved_packing.trees.iter()
                .map(|p| Tree::from_placed(p))
                .collect();

            // Extra local search refinement
            local_search(&mut trees, self.local_iters * 2);
            center_trees(&mut trees);

            if is_valid(&trees) {
                let side = bbox_side(&trees);
                if side < best_side {
                    best_side = side;
                    let mut p = Packing::new();
                    for t in &trees {
                        p.trees.push(t.to_placed());
                    }
                    best_packing = Some(p);
                }
            }
        }

        best_packing.unwrap_or_else(|| {
            // Fallback to evolved
            let evolved = EvolvedPacker::default();
            let packings = evolved.pack_all(n);
            packings[n - 1].clone()
        })
    }

    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        // Get evolved as baseline first (once, not per-N)
        let evolved = EvolvedPacker::default();
        let evolved_packings = evolved.pack_all(max_n);

        let mut results = Vec::with_capacity(max_n);

        for n in 1..=max_n {
            let evolved_packing = &evolved_packings[n - 1];
            let evolved_side = evolved_packing.side_length();
            let mut best_side = evolved_side;
            let mut best_packing = evolved_packing.clone();

            if n == 1 {
                results.push(best_packing);
                continue;
            }

            // Pattern strategies only work well for small N (data shows n <= 20)
            if n <= 20 {
                // Strategy A: Diamond init → Sparrow explore → Wave compact → Local search
                {
                    let init = diamond_init(n);
                    let mut trees = sparrow_explore(n, init, self.sparrow_iters);
                    if is_valid(&trees) {
                        wave_compaction(&mut trees, self.wave_passes);
                        local_search(&mut trees, self.local_iters);
                        center_trees(&mut trees);

                        if is_valid(&trees) {
                            let side = bbox_side(&trees);
                            if side < best_side {
                                best_side = side;
                                let mut p = Packing::new();
                                for t in &trees { p.trees.push(t.to_placed()); }
                                best_packing = p;
                            }
                        }
                    }
                }

                // Strategy B: Hexagonal init → Sparrow explore → Wave compact → Local search
                {
                    let init = hexagonal_init(n);
                    let mut trees = sparrow_explore(n, init, self.sparrow_iters);
                    if is_valid(&trees) {
                        wave_compaction(&mut trees, self.wave_passes);
                        local_search(&mut trees, self.local_iters);
                        center_trees(&mut trees);

                        if is_valid(&trees) {
                            let side = bbox_side(&trees);
                            if side < best_side {
                                best_side = side;
                                let mut p = Packing::new();
                                for t in &trees { p.trees.push(t.to_placed()); }
                                best_packing = p;
                            }
                        }
                    }
                }
            }

            // Strategy C: Evolved base → Extra local refinement (works for all N)
            {
                let mut trees: Vec<Tree> = evolved_packing.trees.iter()
                    .map(|p| Tree::from_placed(p))
                    .collect();

                // More iterations for larger N
                let iters = if n <= 20 { self.local_iters * 2 } else { self.local_iters * 3 };
                local_search(&mut trees, iters);
                center_trees(&mut trees);

                if is_valid(&trees) {
                    let side = bbox_side(&trees);
                    if side < best_side {
                        best_side = side;
                        let mut p = Packing::new();
                        for t in &trees { p.trees.push(t.to_placed()); }
                        best_packing = p;
                    }
                }
            }

            results.push(best_packing);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined() {
        let packer = CombinedPacker::default();
        let packing = packer.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }
}
