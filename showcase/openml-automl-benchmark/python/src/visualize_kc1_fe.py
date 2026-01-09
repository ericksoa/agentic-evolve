#!/usr/bin/env python3
"""
Visualize KC1 feature engineering evolution results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
VISUALS_DIR = Path(__file__).parent.parent.parent / 'visuals'
VISUALS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / 'kc1_fe_evolution_log.json') as f:
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
    ax.axhline(y=0.463, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Previous Best (no FE): 0.463')
    ax.axhline(y=0.501, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='Auto-sklearn Target: 0.501')

    # Best result annotation
    best_idx = np.argmax(holdout_f1)
    ax.annotate(f'Best: {holdout_f1[best_idx]:.4f}\n(Ratios+Bins+Ens)',
                xy=(best_idx + width/2, holdout_f1[best_idx]),
                xytext=(best_idx - 2, holdout_f1[best_idx] + 0.03),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('KC1 Feature Engineering Evolution: 15 Generations\n(Minimal improvement - dataset may be at information limit)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}\n{n[:10]}" for g, n in zip(gens, names)],
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0.35, 0.55)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'kc1_fe_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'kc1_fe_evolution.png'}")


def create_gap_analysis():
    """Analyze overfitting gaps - KC1 shows negative gaps (good generalization)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    gens = [e['generation'] for e in evolution_log]
    gaps = [e['gap'] for e in evolution_log]
    names = [e['name'] for e in evolution_log]
    accepted = [e['accepted'] for e in evolution_log]

    # Color by gap sign: negative = good, positive = overfit
    colors = ['#2ecc71' if g < 0 else ('#e74c3c' if g > 0.05 else '#f39c12')
              for g in gaps]

    bars = ax.barh(range(len(gaps)), gaps, color=colors, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='No gap (perfect)')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Overfit threshold')

    # Mark accepted
    for i, acc in enumerate(accepted):
        if acc:
            ax.scatter([gaps[i]], [i], color='gold', s=200, marker='*',
                      edgecolors='black', linewidth=2, zorder=5)

    ax.set_yticks(range(len(gaps)))
    ax.set_yticklabels([f"G{g}: {n}" for g, n in zip(gens, names)], fontsize=9)
    ax.set_xlabel('CV - Holdout Gap (negative = generalizes well)', fontsize=12)
    ax.set_title('KC1: Overfitting Analysis\n(Green = negative gap, better than CV suggests!)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'kc1_fe_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'kc1_fe_gap_analysis.png'}")


def create_diabetes_vs_kc1_fe_comparison():
    """Compare feature engineering impact on both datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Improvement comparison
    datasets = ['Diabetes', 'KC1']
    no_fe = [0.624, 0.463]
    with_fe = [0.665, 0.464]
    improvements = [6.6, 0.2]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax1.bar(x - width/2, no_fe, width, label='Without FE', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, with_fe, width, label='With FE', color='#2ecc71', alpha=0.8)

    # Add improvement labels
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        height = bar.get_height()
        ax1.annotate(f'+{imp}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='green' if imp > 1 else 'gray')

    ax1.set_ylabel('Holdout F1 Score', fontsize=12)
    ax1.set_title('Feature Engineering Impact\n(Diabetes vs KC1)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Why the difference?
    ax2.axis('off')
    explanation = """
    WHY FEATURE ENGINEERING HELPED DIABETES BUT NOT KC1?
    =====================================================

    DIABETES (Success: +6.6%)
    -------------------------
    • Small dataset (768 samples)
    • Few features (8)
    • Domain features capture medical knowledge
    • Discretization (bins) reduced noise
    • Room for improvement existed

    KC1 (Minimal: +0.2%)
    --------------------
    • Larger dataset (2109 samples)
    • Many features (21) - already rich
    • Software metrics already well-defined
    • XGBoost captures nonlinear patterns
    • Near information-theoretic limit?

    KEY INSIGHT
    -----------
    Feature engineering helps most when:
    1. Original features are insufficient
    2. Domain knowledge adds real signal
    3. Dataset is small enough to benefit
       from regularization via discretization

    KC1 may already be extracting maximum
    signal with standard ML approaches.
    """
    ax2.text(0.1, 0.95, explanation, fontsize=10, family='monospace',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_vs_kc1_fe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_vs_kc1_fe.png'}")


def create_key_insight():
    """Summary insight graphic."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'KC1 Feature Engineering: Key Insights', fontsize=22, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Main finding
    ax.text(0.5, 0.85, 'Minimal Improvement (+0.2%) - Dataset at Limit?',
            fontsize=16, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Left: Results
    results_text = """
    EVOLUTION RESULTS
    =================
    Generations:     15
    Accepted:        2 (Gen 0, 12)

    Best Strategy:   Ratios + Bins + LR+RF
    Best Holdout:    0.4641
    Previous Best:   0.4629
    Improvement:     +0.2%

    Key Finding:
    XGBoost baseline was nearly
    optimal from the start.
    """
    ax.text(0.15, 0.55, results_text, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Middle: What we tried
    tried_text = """
    FEATURE STRATEGIES TESTED
    =========================

    1. Ratio Features      → 0.423 (worse)
    2. Complexity Features → 0.439 (worse)
    3. Quality Features    → 0.421 (worse)
    4. Interaction Features→ 0.411 (worse)
    5. Binned Features     → 0.434 (worse)
    6. Ratios + Complexity → 0.454 (close)
    7. Ratios + Bins       → 0.414 (worse)
    8. All Features        → 0.431 (worse)
    9. Ratios+Cmplx + XGB  → 0.458 (close)
    10. Cmplx+Bins + RF    → 0.446 (worse)
    12. Ratios+Bins + Ens  → 0.464 (BEST)
    """
    ax.text(0.45, 0.55, tried_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

    # Right: Interpretation
    interpret_text = """
    INTERPRETATION
    ==============

    Why didn't FE help KC1?

    1. SOFTWARE METRICS ARE
       ALREADY WELL-ENGINEERED
       • LOC, complexity, Halstead
         metrics are domain features
       • Adding more may be redundant

    2. XGBOOST HANDLES NONLINEARITY
       • Tree models find interactions
       • Manual interactions redundant

    3. IMBALANCE IS THE REAL ISSUE
       • 15.5% defective rate
       • Feature engineering doesn't
         add more minority samples

    4. INFORMATION LIMIT
       • With 21 features & 326 defects,
         may be near theoretical max

    TARGET GAP REMAINS
    ------------------
    Our Best:     0.464
    Auto-sklearn: 0.501 (-7.4%)
    """
    ax.text(0.75, 0.55, interpret_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Bottom: Key takeaway
    ax.text(0.5, 0.08,
            'KEY TAKEAWAY: Feature engineering is dataset-dependent. Diabetes improved +6.6%, KC1 only +0.2%.\n'
            'When original features are already well-engineered (like software metrics), FE provides diminishing returns.',
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(VISUALS_DIR / 'kc1_fe_key_insight.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'kc1_fe_key_insight.png'}")


if __name__ == '__main__':
    print("Creating KC1 feature engineering visualizations...")
    create_fe_evolution_progress()
    create_gap_analysis()
    create_diabetes_vs_kc1_fe_comparison()
    create_key_insight()
    print("\nAll KC1 feature engineering visualizations created!")
