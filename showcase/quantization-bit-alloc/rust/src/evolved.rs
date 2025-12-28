//! Evolved bit allocation heuristic
//!
//! This file is modified by the evolution process.
//! The goal is to find a heuristic that:
//! 1. Maximizes compression (lower avg bits)
//! 2. Maintains accuracy above threshold
//!
//! FINAL CHAMPION: wider_fp16_zone (Gen5)
//! Strategy: Sensitivity-based FP16/FP32 with edge protection
//! - FP32 for LayerNorm, Classifier (always protected)
//! - FP32 for first/last 7.5% of layers (edge protection)
//! - Middle 85%: If sensitivity_at(FP16) > 0.001: use FP32, Else: use FP16
//!
//! TEST fitness: 3.8273 | TEST accuracy: 99.8% | Compression: 1.94x
//! Improvement over Gen1 baseline: 3993%

use crate::{BitAllocationHeuristic, BitWidth, LayerInfo, LayerType};

pub struct Evolved;

impl BitAllocationHeuristic for Evolved {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        // wider_fp16_zone strategy:
        // - Always protect LayerNorm and Classifier with FP32
        // - Protect first/last 7.5% of layers with FP32
        // - For middle 85%, use sensitivity-based allocation:
        //   - High FP16 sensitivity (> 0.001): FP32
        //   - Otherwise: FP16

        // Always protect LayerNorm and Classifier with FP32
        match layer.layer_type {
            LayerType::LayerNorm | LayerType::Classifier => return BitWidth::FP32,
            _ => {}
        }

        // Position-based protection for edges
        let pos = layer.relative_position();

        // First/last 7.5% of layers: protect with FP32
        if pos < 0.075 || pos > 0.925 {
            return BitWidth::FP32;
        }

        // Middle 85%: sensitivity-based allocation
        if layer.sensitivity_at(BitWidth::FP16) > 0.001 {
            // Layer is sensitive at FP16 - use FP32
            BitWidth::FP32
        } else {
            // Default to FP16 for low sensitivity layers
            BitWidth::FP16
        }
    }
}
