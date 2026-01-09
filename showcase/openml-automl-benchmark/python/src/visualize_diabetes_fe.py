#!/usr/bin/env python3
"""
Visualize diabetes feature engineering evolution results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
VISUALS_DIR = Path(__file__).parent.parent.parent / 'visuals'
VISUALS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / 'diabetes_fe_evolution_log.json') as f:
    evolution_log = json.load(f)


def create_fe_evolution_progress():
    """Create feature engineering evolution progress chart."""
    fig, ax = plt.subplots(figsize=(16, 8))

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
    ax.axhline(y=0.624, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Previous Best (no FE): 0.624')
    ax.axhline(y=0.745, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='Auto-sklearn Target: 0.745')

    # Best result annotation
    best_idx = np.argmax(holdout_f1)
    ax.annotate(f'NEW BEST: {holdout_f1[best_idx]:.4f}\n(Domain + Bins)',
                xy=(best_idx + width/2, holdout_f1[best_idx]),
                xytext=(best_idx + 2, holdout_f1[best_idx] + 0.02),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Diabetes Feature Engineering Evolution: 15 Generations\n(Stars = Accepted Improvements)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}\n{n[:10]}" for g, n in zip(gens, names)],
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0.5, 0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_fe_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_fe_evolution.png'}")


def create_feature_count_analysis():
    """Show relationship between feature count and performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    gens = [e['generation'] for e in evolution_log]
    holdout_f1 = [e['holdout_f1'] for e in evolution_log]
    gaps = [e['gap'] for e in evolution_log]
    n_features = [e['params'].get('n_features', 8) for e in evolution_log]
    names = [e['name'] for e in evolution_log]
    accepted = [e['accepted'] for e in evolution_log]

    # Left: Feature count vs holdout F1
    colors = ['green' if a else 'gray' for a in accepted]
    scatter = ax1.scatter(n_features, holdout_f1, c=colors, s=150, edgecolors='black', alpha=0.8)

    # Label key points
    best_idx = np.argmax(holdout_f1)
    ax1.annotate(f'Winner\n(Domain+Bins)',
                xy=(n_features[best_idx], holdout_f1[best_idx]),
                xytext=(n_features[best_idx] + 3, holdout_f1[best_idx]),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Worst point
    worst_idx = np.argmin(holdout_f1)
    ax1.annotate(f'Worst\n({names[worst_idx]})',
                xy=(n_features[worst_idx], holdout_f1[worst_idx]),
                xytext=(n_features[worst_idx] + 3, holdout_f1[worst_idx] - 0.02),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('Holdout F1', fontsize=12)
    ax1.set_title('Feature Count vs Performance\n(More features != Better)', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.624, color='gray', linestyle='--', label='Previous best', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Feature count vs overfitting gap
    colors2 = ['red' if g > 0.08 else 'orange' if g > 0.05 else 'green' for g in gaps]
    ax2.scatter(n_features, gaps, c=colors2, s=150, edgecolors='black', alpha=0.8)

    ax2.axhline(y=0.05, color='orange', linestyle='--', label='Moderate gap', alpha=0.7)
    ax2.axhline(y=0.08, color='red', linestyle='--', label='High gap (overfitting)', alpha=0.7)

    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('CV - Holdout Gap', fontsize=12)
    ax2.set_title('Feature Count vs Overfitting Gap\n(More features = More overfitting risk)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_fe_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_fe_feature_analysis.png'}")


def create_before_after_comparison():
    """Compare before and after feature engineering."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['No FE\n(LR+RF)', 'Ratios', 'Interactions', 'Domain+Bins\n(WINNER)']
    holdout_scores = [0.624, 0.6343, 0.6372, 0.6649]
    improvements = [0, 1.7, 2.1, 6.6]  # % improvement over no-FE

    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    bars = ax.bar(methods, holdout_scores, color=colors, edgecolor='black', linewidth=2)

    # Add improvement labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        label = f'+{imp:.1f}%' if imp > 0 else 'Baseline'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.624, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Previous best (no FE)')
    ax.axhline(y=0.745, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Auto-sklearn target')

    ax.set_ylabel('Holdout F1 Score', fontsize=12)
    ax.set_title('Feature Engineering Impact on Diabetes Dataset\n(Holdout validation shows true improvement)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.55, 0.8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_fe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_fe_comparison.png'}")


def create_key_insight_fe():
    """Create summary insight graphic for feature engineering."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Feature Engineering Evolution: Key Insights', fontsize=22, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Main finding
    ax.text(0.5, 0.85, 'Domain + Bins = +6.6% Improvement!',
            fontsize=18, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # Left box: What worked
    worked_text = """
    WHAT WORKED
    ===========

    1. Domain + Bins (Winner)
       Holdout: 0.665 (+6.6%)
       Gap: 0.037 (minimal overfit)

    2. Interaction Features
       Holdout: 0.637 (+2.1%)
       Simple feature products

    3. Ratio Features
       Holdout: 0.634 (+1.7%)
       Glucose/insulin, BMI/age

    Common pattern:
    - Moderate feature expansion
    - Low CV-holdout gap
    """
    ax.text(0.18, 0.55, worked_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Middle box: What didn't work
    didnt_text = """
    WHAT DIDN'T WORK
    ================

    1. Domain Features Alone
       CV: 0.717, Holdout: 0.609
       Gap: 0.108 (massive overfit!)

    2. All Features Combined
       CV: 0.711, Holdout: 0.628
       Gap: 0.083 (too complex)

    3. Domain + XGBoost
       CV: 0.673, Holdout: 0.544
       Gap: 0.128 (worst overfit!)

    Common pattern:
    - High feature count
    - Large CV-holdout gap
    - Complex models on complex features
    """
    ax.text(0.48, 0.55, didnt_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Right box: Progress
    progress_text = """
    PROGRESS TRACKER
    ================

    Target Comparison:
    ------------------
    Our Best:      0.665
    Auto-sklearn:  0.745 (need 12%)
    FLAML:         0.758 (need 14%)
    AutoGluon:     0.797 (need 20%)

    Journey So Far:
    ---------------
    RF Default:    0.595 (start)
    LR+RF:         0.624 (+4.9%)
    Domain+Bins:   0.665 (+11.8%)

    Gap to Auto-sklearn:
    0.665 / 0.745 = 89.3%
    """
    ax.text(0.78, 0.55, progress_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Bottom: Key takeaway
    ax.text(0.5, 0.08,
            'KEY TAKEAWAY: Feature engineering with BINS (discretization) worked better than complex domain features.\n'
            'The winning combination used LogReg with categorical bins - simpler model on simpler features.',
            fontsize=12, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    plt.savefig(VISUALS_DIR / 'diabetes_fe_key_insight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_fe_key_insight.png'}")


def create_gap_analysis():
    """Analyze overfitting gaps across generations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    gens = [e['generation'] for e in evolution_log]
    gaps = [e['gap'] for e in evolution_log]
    names = [e['name'] for e in evolution_log]
    accepted = [e['accepted'] for e in evolution_log]

    colors = ['#2ecc71' if a else ('#e74c3c' if g > 0.08 else '#f39c12')
              for a, g in zip(accepted, gaps)]

    bars = ax.barh(range(len(gaps)), gaps, color=colors, edgecolor='black')

    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Moderate overfit threshold')
    ax.axvline(x=0.08, color='red', linestyle='--', linewidth=2, label='High overfit threshold')

    ax.set_yticks(range(len(gaps)))
    ax.set_yticklabels([f"G{g}: {n}" for g, n in zip(gens, names)], fontsize=9)
    ax.set_xlabel('CV - Holdout Gap (smaller = better)', fontsize=12)
    ax.set_title('Overfitting Detection by Feature Strategy\n(Green = Accepted, Red = High overfit, Orange = Rejected)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_fe_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_fe_gap_analysis.png'}")


if __name__ == '__main__':
    print("Creating feature engineering visualizations...")
    create_fe_evolution_progress()
    create_feature_count_analysis()
    create_before_after_comparison()
    create_key_insight_fe()
    create_gap_analysis()
    print("\nAll feature engineering visualizations created!")
