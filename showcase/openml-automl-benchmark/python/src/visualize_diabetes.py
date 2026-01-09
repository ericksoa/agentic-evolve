#!/usr/bin/env python3
"""
Visualize diabetes evolution results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
VISUALS_DIR = Path(__file__).parent.parent.parent / 'visuals'
VISUALS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / 'diabetes_evolution_log.json') as f:
    evolution_log = json.load(f)


def create_evolution_progress():
    """Create evolution progress chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    gens = [e['generation'] for e in evolution_log]
    cv_f1 = [e['cv_f1'] for e in evolution_log]
    holdout_f1 = [e['holdout_f1'] for e in evolution_log]
    names = [e['name'] for e in evolution_log]
    accepted = [e['accepted'] for e in evolution_log]

    x = np.arange(len(gens))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, cv_f1, width, label='CV F1', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, holdout_f1, width, label='Holdout F1',
                   color='#2ecc71', alpha=0.8)

    # Mark accepted generations
    for i, acc in enumerate(accepted):
        if acc:
            ax.scatter([i + width/2], [holdout_f1[i]], color='gold', s=150, zorder=5,
                      edgecolors='black', linewidth=2, marker='*')

    # Target lines
    ax.axhline(y=0.648, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Original Baseline (0.648)')
    ax.axhline(y=0.745, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='Auto-sklearn Target (0.745)')

    # Best result annotation
    best_idx = np.argmax(holdout_f1)
    ax.annotate(f'Best: {holdout_f1[best_idx]:.3f}\n(LR+RF Ensemble)',
                xy=(best_idx + width/2, holdout_f1[best_idx]),
                xytext=(best_idx + 2, holdout_f1[best_idx] + 0.03),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Diabetes Evolution: 15 Generations\n(Stars = Accepted Improvements)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}\n{n[:8]}" for g, n in zip(gens, names)],
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0.5, 0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_evolution_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_evolution_progress.png'}")


def create_cv_holdout_comparison():
    """Show the gap between CV and holdout scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cv_f1 = [e['cv_f1'] for e in evolution_log]
    holdout_f1 = [e['holdout_f1'] for e in evolution_log]
    gaps = [e['gap'] for e in evolution_log]
    names = [e['name'] for e in evolution_log]
    gens = [e['generation'] for e in evolution_log]

    # Left: Scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(gens)))
    scatter = ax1.scatter(cv_f1, holdout_f1, c=gens, cmap='viridis',
                         s=150, edgecolors='black', linewidth=1.5, zorder=5)

    # Diagonal
    ax1.plot([0.55, 0.75], [0.55, 0.75], 'k--', alpha=0.5, label='Perfect (no gap)')

    # Label best
    best_idx = np.argmax(holdout_f1)
    ax1.annotate(f'Best: Gen {gens[best_idx]}\n({names[best_idx]})',
                xy=(cv_f1[best_idx], holdout_f1[best_idx]),
                xytext=(cv_f1[best_idx] - 0.05, holdout_f1[best_idx] + 0.02),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

    ax1.set_xlabel('CV F1', fontsize=12)
    ax1.set_ylabel('Holdout F1', fontsize=12)
    ax1.set_title('CV vs Holdout: All Points Below Diagonal\n(Every model overfits somewhat)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Generation')

    # Right: Gap analysis
    colors = ['#e74c3c' if g > 0.08 else '#f39c12' if g > 0.05 else '#2ecc71' for g in gaps]
    ax2.barh(range(len(gaps)), gaps, color=colors, edgecolor='black')
    ax2.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Moderate Gap')
    ax2.axvline(x=0.08, color='red', linestyle='--', linewidth=2, label='High Gap')

    ax2.set_yticks(range(len(gaps)))
    ax2.set_yticklabels([f"G{g}: {n[:10]}" for g, n in zip(gens, names)], fontsize=8)
    ax2.set_xlabel('CV - Holdout Gap', fontsize=12)
    ax2.set_title('Overfitting Gap by Generation\n(Smaller is better)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_cv_vs_holdout.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_cv_vs_holdout.png'}")


def create_key_insight():
    """Create summary insight graphic."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Diabetes Evolution: Key Insights', fontsize=20, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Main finding
    ax.text(0.5, 0.82, 'Simple Ensemble (LR+RF) Won',
            fontsize=16, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Left box: Results
    results_text = """
    EVOLUTION RESULTS
    =================
    Generations:  15
    Accepted:     3 (Gen 0, 1, 11)

    Best Model:   LR + RF Ensemble
    Holdout F1:   0.624

    Key Discovery:
    CV scores (0.65-0.70) overestimate
    true performance by 5-10%
    """
    ax.text(0.22, 0.5, results_text, fontsize=11, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Middle box: What we learned
    lessons_text = """
    MALLORN LESSONS CONFIRMED
    =========================

    1. Simpler ensembles beat complex ones
       - LR+RF > LR+RF+XGB
       - 2 models > 3 models

    2. CV overestimates performance
       - All 15 generations showed
         CV > Holdout (gap 0.01-0.14)

    3. Threshold tuning didn't help
       - t=0.4 and t=0.35 both worse
       - Default 0.5 was better

    4. Class weights sometimes hurt
       - XGBoost with weights: 0.578
       - XGBoost default: 0.594
    """
    ax.text(0.52, 0.5, lessons_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Right box: Comparison
    compare_text = """
    MODEL COMPARISON
    ================
    (Holdout F1)

    LR+RF Ensemble:  0.624 ★
    LogReg Balanced: 0.612
    RF Tuned:        0.604
    GradientBoost:   0.601
    RF Default:      0.595
    XGBoost:         0.594
    SVM RBF:         0.609

    Target Gap:
    Auto-sklearn: 0.745
    Gap: -16.2%
    """
    ax.text(0.82, 0.5, compare_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

    # Bottom: Key takeaway
    ax.text(0.5, 0.08,
            'KEY TAKEAWAY: Proper holdout validation reveals that most "improvements" are illusions.\n'
            'The evolution correctly identified the one strategy that actually helped: simple ensembling.',
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(VISUALS_DIR / 'diabetes_key_insight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_key_insight.png'}")


def create_comparison_chart():
    """Compare KC1 vs Diabetes findings."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Data
    datasets = ['KC1\n(Imbalanced)', 'Diabetes\n(Moderate)']
    baseline_cv = [0.436, 0.648]
    best_holdout = [0.463, 0.624]
    best_cv = [0.430, 0.672]

    x = np.arange(len(datasets))
    width = 0.25

    bars1 = ax.bar(x - width, baseline_cv, width, label='Baseline (CV)', color='#3498db', alpha=0.7)
    bars2 = ax.bar(x, best_cv, width, label='Best Gen (CV)', color='#9b59b6', alpha=0.7)
    bars3 = ax.bar(x + width, best_holdout, width, label='Best Gen (Holdout)', color='#2ecc71', alpha=0.9)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('KC1 vs Diabetes: Evolution Results\n(Holdout validation shows true performance)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3, axis='y')

    # Add insight annotations
    ax.annotate('KC1: Baseline was\nalready optimal',
               xy=(0, 0.463), xytext=(-0.3, 0.55),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('Diabetes: Ensemble\nimproved holdout',
               xy=(1 + width, 0.624), xytext=(1.3, 0.70),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'kc1_vs_diabetes_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'kc1_vs_diabetes_comparison.png'}")


if __name__ == '__main__':
    print("Creating diabetes visualizations...")
    create_evolution_progress()
    create_cv_holdout_comparison()
    create_key_insight()
    create_comparison_chart()
    print("\nAll visualizations created!")
