//! Generate synthetic layer sensitivity data
//!
//! Creates realistic neural network layer profiles with quantization sensitivity.
//! Modeled after transformer architectures (like LLaMA, GPT-2) and ResNets.

use quantization_bit_alloc::{LayerInfo, LayerType};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct Dataset {
    name: String,
    description: String,
    layers: Vec<LayerInfo>,
}

fn main() {
    // Generate three model architectures for train/valid/test
    let train = generate_transformer_model("train", 42, 32, "LLaMA-7B-like");
    let valid = generate_transformer_model("valid", 123, 24, "GPT-2-Medium-like");
    let test = generate_mixed_model("test", 456);

    // Ensure data directory exists
    fs::create_dir_all("data").expect("Failed to create data directory");

    // Write datasets
    fs::write(
        "data/train.json",
        serde_json::to_string_pretty(&train).unwrap(),
    ).expect("Failed to write train.json");

    fs::write(
        "data/valid.json",
        serde_json::to_string_pretty(&valid).unwrap(),
    ).expect("Failed to write valid.json");

    fs::write(
        "data/test.json",
        serde_json::to_string_pretty(&test).unwrap(),
    ).expect("Failed to write test.json");

    println!("Generated datasets:");
    println!("  train.json: {} layers ({})", train.layers.len(), train.description);
    println!("  valid.json: {} layers ({})", valid.layers.len(), valid.description);
    println!("  test.json:  {} layers ({})", test.layers.len(), test.description);
}

/// Generate a transformer-style model
fn generate_transformer_model(name: &str, seed: u64, num_blocks: usize, desc: &str) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut layers = Vec::new();

    // Embedding layer
    layers.push(generate_layer(
        &mut rng,
        0,
        LayerType::Embedding,
        32000 * 4096,  // vocab_size * hidden_dim
        true,  // first layer
        false,
    ));

    // Transformer blocks
    for block in 0..num_blocks {
        let is_early = block < num_blocks / 4;
        let is_late = block > 3 * num_blocks / 4;

        // Self-attention (Q, K, V, O projections)
        layers.push(generate_layer(
            &mut rng,
            layers.len(),
            LayerType::Attention,
            4 * 4096 * 4096,  // 4 projections
            is_early,
            is_late,
        ));

        // LayerNorm after attention
        layers.push(generate_layer(
            &mut rng,
            layers.len(),
            LayerType::LayerNorm,
            4096 * 2,  // gamma + beta
            is_early,
            is_late,
        ));

        // MLP (up projection + down projection)
        layers.push(generate_layer(
            &mut rng,
            layers.len(),
            LayerType::Linear,
            4096 * 11008 * 2,  // FFN hidden dim
            is_early,
            is_late,
        ));

        // LayerNorm after MLP
        layers.push(generate_layer(
            &mut rng,
            layers.len(),
            LayerType::LayerNorm,
            4096 * 2,
            is_early,
            is_late,
        ));
    }

    // Final LayerNorm
    layers.push(generate_layer(
        &mut rng,
        layers.len(),
        LayerType::LayerNorm,
        4096 * 2,
        false,
        true,
    ));

    // Classifier head (LM head)
    layers.push(generate_layer(
        &mut rng,
        layers.len(),
        LayerType::Classifier,
        4096 * 32000,
        false,
        true,  // last layer
    ));

    // Update total params and num_layers
    let total_params: u64 = layers.iter().map(|l| l.num_params).sum();
    let num_layers = layers.len();
    for (i, layer) in layers.iter_mut().enumerate() {
        layer.total_params = total_params;
        layer.num_layers = num_layers;
        layer.layer_idx = i;
    }

    Dataset {
        name: name.to_string(),
        description: desc.to_string(),
        layers,
    }
}

/// Generate a mixed model (ResNet + classifier)
fn generate_mixed_model(name: &str, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut layers = Vec::new();

    // Initial conv
    layers.push(generate_layer(
        &mut rng,
        0,
        LayerType::Conv,
        64 * 3 * 7 * 7,
        true,
        false,
    ));

    // ResNet blocks (simplified)
    let channels = [64, 128, 256, 512];
    for (stage, &ch) in channels.iter().enumerate() {
        let is_early = stage == 0;
        let is_late = stage == channels.len() - 1;

        for block in 0..3 {
            // Conv layers in residual block
            layers.push(generate_layer(
                &mut rng,
                layers.len(),
                LayerType::Conv,
                ch * ch * 3 * 3,
                is_early && block == 0,
                is_late && block == 2,
            ));
            layers.push(generate_layer(
                &mut rng,
                layers.len(),
                LayerType::Conv,
                ch * ch * 3 * 3,
                false,
                false,
            ));
        }
    }

    // Final classifier
    layers.push(generate_layer(
        &mut rng,
        layers.len(),
        LayerType::Linear,
        512 * 1000,
        false,
        true,
    ));

    layers.push(generate_layer(
        &mut rng,
        layers.len(),
        LayerType::Classifier,
        1000,
        false,
        true,
    ));

    // Update total params and num_layers
    let total_params: u64 = layers.iter().map(|l| l.num_params).sum();
    let num_layers = layers.len();
    for (i, layer) in layers.iter_mut().enumerate() {
        layer.total_params = total_params;
        layer.num_layers = num_layers;
        layer.layer_idx = i;
    }

    Dataset {
        name: name.to_string(),
        description: "ResNet-50-like CNN".to_string(),
        layers,
    }
}

/// Generate a single layer with realistic statistics
fn generate_layer(
    rng: &mut ChaCha8Rng,
    idx: usize,
    layer_type: LayerType,
    num_params: u64,
    is_early: bool,
    is_late: bool,
) -> LayerInfo {
    // Weight statistics vary by layer type
    let (weight_mean, weight_std, weight_range) = match layer_type {
        LayerType::Embedding => (0.0, 0.02, 0.1),
        LayerType::LayerNorm => (1.0, 0.01, 0.05),  // Gamma near 1
        LayerType::Classifier => (0.0, 0.01, 0.05),
        _ => (0.0, 0.02 + rng.gen::<f64>() * 0.01, 0.1 + rng.gen::<f64>() * 0.05),
    };

    // Activation range
    let activation_range = match layer_type {
        LayerType::LayerNorm => 1.0 + rng.gen::<f64>() * 0.5,
        LayerType::Classifier => 10.0 + rng.gen::<f64>() * 5.0,
        _ => 2.0 + rng.gen::<f64>() * 3.0,
    };

    // Gradient sensitivity (how much this layer affects loss)
    let base_sensitivity = match layer_type {
        LayerType::LayerNorm => 0.8 + rng.gen::<f64>() * 0.15,  // Very sensitive
        LayerType::Embedding => 0.6 + rng.gen::<f64>() * 0.2,   // Sensitive
        LayerType::Classifier => 0.7 + rng.gen::<f64>() * 0.2,  // Sensitive
        LayerType::Attention => 0.4 + rng.gen::<f64>() * 0.3,   // Moderate
        LayerType::Linear => 0.2 + rng.gen::<f64>() * 0.3,      // Lower
        LayerType::Conv => 0.2 + rng.gen::<f64>() * 0.25,       // Lower
    };

    // Position affects sensitivity
    let position_factor = if is_early { 1.2 } else if is_late { 1.15 } else { 1.0 };
    let gradient_sensitivity = (base_sensitivity * position_factor).min(0.99);

    // Quantization sensitivity at each bit width
    // Higher gradient sensitivity -> higher quantization sensitivity
    let sensitivity = generate_sensitivity_curve(rng, gradient_sensitivity, layer_type);

    LayerInfo {
        layer_idx: idx,
        num_layers: 0,  // Will be updated later
        layer_type,
        num_params,
        total_params: 0,  // Will be updated later
        weight_mean,
        weight_std,
        weight_range,
        activation_range,
        gradient_sensitivity,
        sensitivity,
    }
}

/// Generate quantization sensitivity curve
/// Returns [INT2, INT4, INT8, FP16, FP32] sensitivities
fn generate_sensitivity_curve(
    rng: &mut ChaCha8Rng,
    base_sensitivity: f64,
    layer_type: LayerType,
) -> [f64; 5] {
    // Base curves for different layer types
    // These represent typical accuracy loss at each bit width
    let base_curve = match layer_type {
        LayerType::LayerNorm => [0.50, 0.20, 0.05, 0.001, 0.0],  // Very sensitive
        LayerType::Embedding => [0.40, 0.15, 0.03, 0.001, 0.0],  // Sensitive
        LayerType::Classifier => [0.35, 0.12, 0.02, 0.001, 0.0], // Sensitive
        LayerType::Attention => [0.25, 0.08, 0.015, 0.0005, 0.0], // Moderate
        LayerType::Linear => [0.20, 0.06, 0.01, 0.0002, 0.0],    // Robust
        LayerType::Conv => [0.18, 0.05, 0.008, 0.0001, 0.0],     // Most robust
    };

    // Scale by gradient sensitivity and add proportional noise
    let mut sensitivity = [0.0; 5];
    for i in 0..5 {
        let scale = 0.5 + base_sensitivity * 0.5;  // Scale factor 0.5-1.0
        // Noise proportional to base value (±10% of base, not absolute)
        let noise_scale = base_curve[i] * 0.1;
        let noise = (rng.gen::<f64>() * 2.0 - 1.0) * noise_scale;
        sensitivity[i] = (base_curve[i] * scale + noise).max(0.0).min(0.99);
    }

    // Ensure monotonically decreasing (more bits = less loss)
    for i in (0..4).rev() {
        if sensitivity[i] < sensitivity[i + 1] {
            sensitivity[i] = sensitivity[i + 1] + 0.01;
        }
    }

    sensitivity
}
