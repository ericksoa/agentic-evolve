#!/usr/bin/env python3
"""
Visualize evolution progress for the OpenML benchmark.

Creates charts showing:
1. F1 score progression across generations
2. Comparison with AutoML targets
3. Overfitting detection (CV vs holdout gap)
4. Model complexity vs performance
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# Target thresholds (from AMLB research)
TARGETS = {
    'baseline': {'f1': 0.436, 'label': 'XGBoost Default', 'color': '#888888'},
    'autosklearn': {'f1': 0.501, 'label': 'Auto-sklearn Target (+15%)', 'color': '#FFA500'},
    'flaml': {'f1': 0.510, 'label': 'FLAML Target (+17%)', 'color': '#FF6B6B'},
    'autogluon': {'f1': 0.536, 'label': 'AutoGluon Target (+23%)', 'color': '#4ECDC4'},
}


def create_evolution_chart(
    generations: List[Dict],
    output_path: str = 'evolution_progress.png',
    title: str = 'KC1 Evolution Progress'
):
    """
    Create a chart showing F1 score progression with target lines.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract data
    gen_nums = [g['generation'] for g in generations]
    cv_scores = [g['cv_f1'] for g in generations]
    holdout_scores = [g['holdout_f1'] for g in generations]

    # Plot evolution lines
    ax.plot(gen_nums, cv_scores, 'b-o', linewidth=2, markersize=8,
            label='CV F1 (selection)', alpha=0.7)
    ax.plot(gen_nums, holdout_scores, 'g-s', linewidth=2, markersize=8,
            label='Holdout F1 (true performance)', alpha=0.9)

    # Add target lines
    max_gen = max(gen_nums) if gen_nums else 10
    for key, target in TARGETS.items():
        ax.axhline(y=target['f1'], color=target['color'], linestyle='--',
                   linewidth=2, alpha=0.7, label=target['label'])

    # Styling
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.7)

    # Add annotations for best points
    if holdout_scores:
        best_idx = np.argmax(holdout_scores)
        best_score = holdout_scores[best_idx]
        best_gen = gen_nums[best_idx]
        ax.annotate(f'Best: {best_score:.3f}',
                    xy=(best_gen, best_score),
                    xytext=(best_gen + 0.5, best_score + 0.02),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_overfitting_chart(
    generations: List[Dict],
    output_path: str = 'overfitting_analysis.png'
):
    """
    Create a chart showing the CV-Holdout gap (overfitting indicator).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gen_nums = [g['generation'] for g in generations]
    cv_scores = [g['cv_f1'] for g in generations]
    holdout_scores = [g['holdout_f1'] for g in generations]
    gaps = [g['cv_f1'] - g['holdout_f1'] for g in generations]

    # Left plot: Gap over generations
    colors = ['red' if gap > 0.10 else 'green' for gap in gaps]
    ax1.bar(gen_nums, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.10, color='red', linestyle='--', linewidth=2,
                label='Overfitting Threshold (0.10)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('CV - Holdout Gap', fontsize=12)
    ax1.set_title('Overfitting Detection\n(Red = Overfit, Green = Good)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right plot: CV vs Holdout scatter
    ax2.scatter(cv_scores, holdout_scores, c=gen_nums, cmap='viridis',
                s=100, edgecolors='black', linewidth=1)

    # Add diagonal line (perfect = no overfitting)
    min_val = min(min(cv_scores), min(holdout_scores)) - 0.05
    max_val = max(max(cv_scores), max(holdout_scores)) + 0.05
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
             label='Perfect (no overfit)')

    # Add overfitting zone
    ax2.fill_between([min_val, max_val], [min_val - 0.10, max_val - 0.10],
                     [min_val, max_val], alpha=0.1, color='red',
                     label='Overfitting Zone')

    ax2.set_xlabel('CV F1', fontsize=12)
    ax2.set_ylabel('Holdout F1', fontsize=12)
    ax2.set_title('CV vs Holdout Performance\n(Points below line = overfitting)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Colorbar for generation
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Generation', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_comparison_chart(
    baseline_f1: float,
    evolved_f1: float,
    output_path: str = 'automl_comparison.png'
):
    """
    Create a bar chart comparing our evolved result vs AutoML targets.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    categories = ['XGBoost\nDefault', 'Auto-sklearn\nTarget', 'FLAML\nTarget',
                  'AutoGluon\nTarget', 'evolve-sdk\nResult']
    values = [
        TARGETS['baseline']['f1'],
        TARGETS['autosklearn']['f1'],
        TARGETS['flaml']['f1'],
        TARGETS['autogluon']['f1'],
        evolved_f1
    ]
    colors = ['#888888', '#FFA500', '#FF6B6B', '#4ECDC4', '#2ECC71']

    # Create bars
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    # Add improvement annotation
    improvement = ((evolved_f1 - baseline_f1) / baseline_f1) * 100
    ax.annotate(f'+{improvement:.1f}%\nimprovement',
                xy=(4, evolved_f1),
                xytext=(4.5, evolved_f1 + 0.05),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Styling
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('KC1: evolve-sdk vs AutoML Frameworks\n(Higher is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Add "BEATS" annotations
    if evolved_f1 >= TARGETS['autosklearn']['f1']:
        ax.text(1, TARGETS['autosklearn']['f1'] + 0.01, '✓ BEATEN',
                ha='center', fontsize=9, color='green', fontweight='bold')
    if evolved_f1 >= TARGETS['flaml']['f1']:
        ax.text(2, TARGETS['flaml']['f1'] + 0.01, '✓ BEATEN',
                ha='center', fontsize=9, color='green', fontweight='bold')
    if evolved_f1 >= TARGETS['autogluon']['f1']:
        ax.text(3, TARGETS['autogluon']['f1'] + 0.01, '✓ BEATEN',
                ha='center', fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_baseline_comparison(output_path: str = 'baseline_models.png'):
    """
    Create a chart showing baseline model performance on KC1.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline results from pilot
    models = ['XGBoost', 'LightGBM', 'RandomForest', 'DecisionTree',
              'NaiveBayes', 'KNN', 'GradientBoosting', 'LogisticReg']
    f1_scores = [0.436, 0.436, 0.409, 0.400, 0.395, 0.326, 0.302, 0.298]
    gaps = [0.128, 0.120, 0.130, 0.167, 0.000, 0.038, 0.061, 0.006]

    x = np.arange(len(models))
    width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, gaps, width, label='Train-Test Gap (overfit)',
                   color='#e74c3c', edgecolor='black', alpha=0.7)

    # Styling
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('KC1 Baseline Models: F1 Score vs Overfitting\n(High gap = overfitting)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add insight annotation
    ax.annotate('XGBoost: Best F1 but\nhigh overfitting (0.128 gap)',
                xy=(0, 0.436), xytext=(2, 0.5),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.annotate('LogReg: Low F1 but\nalmost no overfitting',
                xy=(7, 0.298), xytext=(5, 0.15),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_imbalance_explanation(output_path: str = 'class_imbalance.png'):
    """
    Create a visual explanation of class imbalance in KC1.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Pie chart of class distribution
    sizes = [84.5, 15.5]  # Based on 5.47 imbalance ratio
    labels = ['Non-Defective\n(84.5%)', 'Defective\n(15.5%)']
    colors = ['#3498db', '#e74c3c']
    explode = (0, 0.1)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12})
    ax1.set_title('KC1 Class Distribution\n(5.47x Imbalance)', fontsize=14, fontweight='bold')

    # Right: Threshold illustration
    thresholds = np.arange(0.1, 0.9, 0.05)
    # Simulated precision/recall curves for imbalanced data
    precisions = 0.3 + 0.5 * (1 - thresholds) + np.random.normal(0, 0.02, len(thresholds))
    recalls = 0.9 - 0.7 * thresholds + np.random.normal(0, 0.02, len(thresholds))
    f1s = 2 * precisions * recalls / (precisions + recalls + 0.001)

    ax2.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax2.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    ax2.plot(thresholds, f1s, 'g-', linewidth=3, label='F1 Score')

    # Mark optimal threshold
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax2.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.8,
                label=f'Optimal ({best_threshold:.2f})')

    ax2.scatter([best_threshold], [f1s[best_idx]], color='green', s=100, zorder=5)
    ax2.annotate(f'Best F1: {f1s[best_idx]:.2f}',
                 xy=(best_threshold, f1s[best_idx]),
                 xytext=(best_threshold + 0.1, f1s[best_idx] + 0.05),
                 fontsize=10, fontweight='bold')

    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Why Threshold Matters for Imbalanced Data\n(Default 0.5 is rarely optimal)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 0.9)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'visuals'
    output_dir.mkdir(exist_ok=True)

    print("Creating baseline visualizations...")

    # Create explanatory charts
    create_baseline_comparison(str(output_dir / 'kc1_baseline_models.png'))
    create_imbalance_explanation(str(output_dir / 'kc1_class_imbalance.png'))

    # Create placeholder evolution chart with dummy data
    # This will be updated as evolution progresses
    dummy_generations = [
        {'generation': 0, 'cv_f1': 0.436, 'holdout_f1': 0.308},
    ]
    create_evolution_chart(dummy_generations, str(output_dir / 'kc1_evolution_progress.png'))

    print(f"\nVisualizations saved to: {output_dir}")
