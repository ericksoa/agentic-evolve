//! Pattern-based packing using known efficient tree arrangements.
//!
//! Key insight from geometry analysis:
//! - Tree area: 0.2456, much smaller than bounding box (0.7)
//! - 45° rotation gives smallest bbox (0.813 x 0.813)
//! - Current efficiency: 56%, target: 70%
//! - Need better interlocking patterns

use crate::{Packing, PlacedTree};
use rand::Rng;

/// Pre-computed efficient patterns for small N
/// These were discovered through optimization and can be hardcoded

/// For N=2: Two trees with optimal arrangement
fn pattern_n2() -> Vec<(f64, f64, f64)> {
    // Two trees at 45° offset, positioned for minimal bbox
    vec![
        (0.0, 0.0, 0.0),
        (0.55, 0.35, 45.0),
    ]
}

/// For N=3: Triangle arrangement
fn pattern_n3() -> Vec<(f64, f64, f64)> {
    vec![
        (0.0, 0.0, 0.0),
        (0.6, 0.3, 45.0),
        (0.3, 0.8, 180.0),
    ]
}

/// For N=4: Quad arrangement optimized for square bbox
fn pattern_n4() -> Vec<(f64, f64, f64)> {
    vec![
        (0.0, 0.0, 45.0),
        (0.7, 0.0, 315.0),
        (0.0, 0.7, 135.0),
        (0.7, 0.7, 225.0),
    ]
}

/// Herringbone pattern for larger N
/// Trees alternate between 45° and 225° in a grid pattern
fn herringbone_pattern(n: usize) -> Vec<(f64, f64, f64)> {
    let mut trees = Vec::with_capacity(n);

    // Optimal spacing for 45° trees: bbox is 0.813x0.813
    // Add some margin for non-overlap: ~0.85
    let spacing = 0.85;

    let cols = ((n as f64).sqrt().ceil() as usize).max(1);
    let rows = ((n as f64 / cols as f64).ceil() as usize).max(1);

    let mut count = 0;
    for row in 0..rows {
        for col in 0..cols {
            if count >= n {
                break;
            }

            let x = col as f64 * spacing;
            let y = row as f64 * spacing;

            // Alternate rotation in checkerboard pattern
            let angle = if (row + col) % 2 == 0 { 45.0 } else { 225.0 };

            trees.push((x, y, angle));
            count += 1;
        }
    }

    trees
}

/// Diamond pattern: trees at 45° in a tighter arrangement
fn diamond_pattern(n: usize) -> Vec<(f64, f64, f64)> {
    let mut trees = Vec::with_capacity(n);

    // Stagger rows for tighter packing
    let h_spacing = 0.80;
    let v_spacing = 0.75;
    let offset = 0.40; // Offset for alternate rows

    let cols = ((n as f64).sqrt().ceil() as usize).max(1);
    let rows = ((n as f64 / cols as f64).ceil() as usize).max(1);

    let mut count = 0;
    for row in 0..rows {
        for col in 0..cols {
            if count >= n {
                break;
            }

            let row_offset = if row % 2 == 0 { 0.0 } else { offset };
            let x = col as f64 * h_spacing + row_offset;
            let y = row as f64 * v_spacing;

            // All trees at 45° for consistent smaller bbox
            let angle = 45.0;

            trees.push((x, y, angle));
            count += 1;
        }
    }

    trees
}

/// Spiral pattern with optimized rotations
fn spiral_pattern(n: usize) -> Vec<(f64, f64, f64)> {
    let mut trees = Vec::with_capacity(n);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    for i in 0..n {
        let angle = i as f64 * golden_angle;
        let dist = 0.45 + (i as f64 * 0.35).sqrt();

        let x = dist * angle.cos();
        let y = dist * angle.sin();

        // Rotate tree to point outward
        let tree_angle = ((angle.to_degrees() / 45.0).round() * 45.0 + 45.0) % 360.0;

        trees.push((x, y, tree_angle));
    }

    trees
}

/// Validate and repair a pattern by adjusting positions
fn validate_and_repair(
    pattern: &mut Vec<(f64, f64, f64)>,
    max_iterations: usize,
) -> bool {
    let mut rng = rand::thread_rng();

    for _ in 0..max_iterations {
        // Check for overlaps
        let trees: Vec<PlacedTree> = pattern
            .iter()
            .map(|(x, y, a)| PlacedTree::new(*x, *y, *a))
            .collect();

        let mut has_overlap = false;
        for i in 0..trees.len() {
            for j in (i + 1)..trees.len() {
                if trees[i].overlaps(&trees[j]) {
                    has_overlap = true;

                    // Push trees apart
                    let dx = pattern[j].0 - pattern[i].0;
                    let dy = pattern[j].1 - pattern[i].1;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    let push = 0.05 + rng.gen::<f64>() * 0.02;
                    pattern[i].0 -= push * dx / dist;
                    pattern[i].1 -= push * dy / dist;
                    pattern[j].0 += push * dx / dist;
                    pattern[j].1 += push * dy / dist;

                    // Sometimes try rotation change
                    if rng.gen::<f64>() < 0.1 {
                        let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
                        pattern[i].2 = angles[rng.gen_range(0..8)];
                        pattern[j].2 = angles[rng.gen_range(0..8)];
                    }
                }
            }
        }

        if !has_overlap {
            return true;
        }
    }

    // Final check
    let trees: Vec<PlacedTree> = pattern
        .iter()
        .map(|(x, y, a)| PlacedTree::new(*x, *y, *a))
        .collect();

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if trees[i].overlaps(&trees[j]) {
                return false;
            }
        }
    }
    true
}

/// Compact pattern toward center
fn compact_pattern(pattern: &mut Vec<(f64, f64, f64)>, iterations: usize) {
    for _ in 0..iterations {
        if pattern.is_empty() {
            return;
        }

        // Find center
        let cx: f64 = pattern.iter().map(|p| p.0).sum::<f64>() / pattern.len() as f64;
        let cy: f64 = pattern.iter().map(|p| p.1).sum::<f64>() / pattern.len() as f64;

        for i in 0..pattern.len() {
            let (x, y, angle) = pattern[i];
            let dx = cx - x;
            let dy = cy - y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.01 {
                let new_x = x + dx * 0.02;
                let new_y = y + dy * 0.02;

                // Test if this creates overlap
                let mut test = pattern.clone();
                test[i] = (new_x, new_y, angle);

                let trees: Vec<PlacedTree> = test
                    .iter()
                    .map(|(px, py, pa)| PlacedTree::new(*px, *py, *pa))
                    .collect();

                let mut ok = true;
                for j in 0..trees.len() {
                    if i != j && trees[i].overlaps(&trees[j]) {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    pattern[i] = (new_x, new_y, angle);
                }
            }
        }
    }
}

/// Center pattern at origin
fn center_pattern(pattern: &mut Vec<(f64, f64, f64)>) {
    if pattern.is_empty() {
        return;
    }

    let trees: Vec<PlacedTree> = pattern
        .iter()
        .map(|(x, y, a)| PlacedTree::new(*x, *y, *a))
        .collect();

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in &trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    for p in pattern.iter_mut() {
        p.0 -= center_x;
        p.1 -= center_y;
    }
}

/// Calculate bounding box side length
fn pattern_side(pattern: &[(f64, f64, f64)]) -> f64 {
    if pattern.is_empty() {
        return 0.0;
    }

    let trees: Vec<PlacedTree> = pattern
        .iter()
        .map(|(x, y, a)| PlacedTree::new(*x, *y, *a))
        .collect();

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in &trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

pub struct PatternBasedPacker;

impl PatternBasedPacker {
    pub fn pack(&self, n: usize) -> Packing {
        // Try multiple patterns and pick the best
        let patterns: Vec<Vec<(f64, f64, f64)>> = vec![
            if n == 2 { pattern_n2() } else if n == 3 { pattern_n3() } else if n == 4 { pattern_n4() } else { herringbone_pattern(n) },
            diamond_pattern(n),
            spiral_pattern(n),
        ];

        let mut best_pattern = None;
        let mut best_side = f64::INFINITY;

        for mut pattern in patterns {
            if validate_and_repair(&mut pattern, 500) {
                compact_pattern(&mut pattern, 100);
                center_pattern(&mut pattern);

                let side = pattern_side(&pattern);
                if side < best_side {
                    best_side = side;
                    best_pattern = Some(pattern);
                }
            }
        }

        let pattern = best_pattern.unwrap_or_else(|| {
            let mut p = spiral_pattern(n);
            validate_and_repair(&mut p, 1000);
            p
        });

        let mut packing = Packing::new();
        for (x, y, angle) in pattern {
            packing.trees.push(PlacedTree::new(x, y, angle));
        }
        packing
    }

    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        (1..=max_n).map(|n| self.pack(n)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_packer() {
        let packer = PatternBasedPacker;
        let packing = packer.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }
}
