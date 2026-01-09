#!/usr/bin/env python3
"""
Comprehensive visualization of all diabetes improvement experiments.
Creates charts explaining the who, what, and why of each approach.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
VISUALS_DIR = Path(__file__).parent.parent.parent.parent / 'visuals'
VISUALS_DIR.mkdir(exist_ok=True)

# Load all experiment results
def load_results():
    results = {}
    for exp_file in RESULTS_DIR.glob('exp_*.json'):
        exp_name = exp_file.stem.replace('exp_', '').replace('_results', '')
        with open(exp_file) as f:
            results[exp_name] = json.load(f)
    return results


def create_master_comparison():
    """Create the main comparison chart of all experiments."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Data from experiments
    experiments = {
        'Baseline\n(Domain+Bins)': {'holdout': 0.665, 'beat': False, 'color': '#95a5a6'},
        'SMOTE\n(k=5)': {'holdout': 0.677, 'beat': True, 'color': '#3498db'},
        'Stacking\n(4 models)': {'holdout': 0.644, 'beat': False, 'color': '#e74c3c'},
        'Feature\nSelection': {'holdout': 0.690, 'beat': True, 'color': '#2ecc71'},
        'Binning\n(4 quantile)': {'holdout': 0.666, 'beat': True, 'color': '#9b59b6'},
        'MLP\n(128,64)': {'holdout': 0.665, 'beat': False, 'color': '#e67e22'},
        'Calibration\n(Sigmoid)': {'holdout': 0.675, 'beat': True, 'color': '#1abc9c'},
        'Regularization\n(C=0.5)': {'holdout': 0.673, 'beat': True, 'color': '#f39c12'},
    }

    names = list(experiments.keys())
    holdouts = [experiments[n]['holdout'] for n in names]
    colors = [experiments[n]['color'] for n in names]
    beats = [experiments[n]['beat'] for n in names]

    # Sort by holdout score
    sorted_idx = np.argsort(holdouts)[::-1]
    names = [names[i] for i in sorted_idx]
    holdouts = [holdouts[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    beats = [beats[i] for i in sorted_idx]

    x = np.arange(len(names))
    bars = ax.bar(x, holdouts, color=colors, edgecolor='black', linewidth=2)

    # Add stars for experiments that beat baseline
    for i, (bar, beat) in enumerate(zip(bars, beats)):
        if beat:
            ax.scatter([i], [bar.get_height() + 0.008], marker='*', s=200,
                      color='gold', edgecolors='black', linewidth=1, zorder=5)

    # Reference lines
    ax.axhline(y=0.665, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Baseline (0.665)')
    ax.axhline(y=0.745, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Auto-sklearn Target (0.745)')

    # Labels
    for i, (bar, h) in enumerate(zip(bars, holdouts)):
        ax.annotate(f'{h:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, -15), textcoords="offset points",
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color='white')

    ax.set_ylabel('Holdout F1 Score', fontsize=12)
    ax.set_title('Diabetes Improvement Experiments: All 8 Strategies Compared\n(Stars = Beat 0.665 Baseline)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0.6, 0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_all_experiments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_all_experiments.png'}")


def create_improvement_waterfall():
    """Create waterfall chart showing cumulative improvements."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Progression of improvements
    stages = [
        ('RF Baseline', 0.595, 0),
        ('+ Ensemble (LR+RF)', 0.624, 0.029),
        ('+ Domain Features', 0.637, 0.013),
        ('+ Binning', 0.665, 0.028),
        ('+ Feature Selection', 0.690, 0.025),
    ]

    names = [s[0] for s in stages]
    values = [s[1] for s in stages]
    deltas = [s[2] for s in stages]

    x = np.arange(len(names))

    # Create waterfall
    colors = ['#3498db'] + ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas[1:]]
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=2)

    # Add delta labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        if i > 0:
            ax.annotate(f'+{delta:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color='green')

    # Target line
    ax.axhline(y=0.745, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Auto-sklearn Target (0.745)')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, -15), textcoords="offset points",
                   ha='center', va='top', fontsize=11, fontweight='bold',
                   color='white')

    ax.set_ylabel('Holdout F1 Score', fontsize=12)
    ax.set_title('Journey from Baseline to Best: Cumulative Improvements\n(Each bar shows the technique that helped most)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.5, 0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_improvement_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_improvement_waterfall.png'}")


def create_what_worked_didnt():
    """Create side-by-side comparison of what worked vs didn't."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # What worked
    worked = [
        ('Feature Selection\n(RFE, 10 features)', 0.690, '+3.7%'),
        ('SMOTE\n(k=5 neighbors)', 0.677, '+1.8%'),
        ('Calibration\n(Sigmoid + t=0.4)', 0.675, '+1.5%'),
        ('Regularization\n(LogReg C=0.5)', 0.673, '+1.2%'),
        ('Binning Tuning\n(4 quantile bins)', 0.666, '+0.2%'),
    ]

    names1 = [w[0] for w in worked]
    vals1 = [w[1] for w in worked]
    pcts1 = [w[2] for w in worked]

    y1 = np.arange(len(names1))
    bars1 = ax1.barh(y1, vals1, color='#2ecc71', edgecolor='black', linewidth=2)
    ax1.axvline(x=0.665, color='gray', linestyle='--', linewidth=2, label='Baseline')

    for i, (bar, pct) in enumerate(zip(bars1, pcts1)):
        ax1.annotate(pct, xy=(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10, fontweight='bold', color='green')

    ax1.set_yticks(y1)
    ax1.set_yticklabels(names1, fontsize=10)
    ax1.set_xlabel('Holdout F1', fontsize=11)
    ax1.set_title('WHAT WORKED', fontsize=14, fontweight='bold', color='green')
    ax1.set_xlim(0.63, 0.72)
    ax1.legend(loc='lower right')
    ax1.invert_yaxis()

    # What didn't work
    didnt = [
        ('Stacking\n(4 base models)', 0.644, '-3.2%'),
        ('MLP Neural Net\n(128,64 tanh)', 0.665, '0%'),
        ('All Features\n(no selection)', 0.665, '0%'),
        ('BorderlineSMOTE', 0.654, '-1.7%'),
        ('Isotonic Calibration', 0.644, '-3.2%'),
    ]

    names2 = [d[0] for d in didnt]
    vals2 = [d[1] for d in didnt]
    pcts2 = [d[2] for d in didnt]

    y2 = np.arange(len(names2))
    bars2 = ax2.barh(y2, vals2, color='#e74c3c', edgecolor='black', linewidth=2)
    ax2.axvline(x=0.665, color='gray', linestyle='--', linewidth=2, label='Baseline')

    for i, (bar, pct) in enumerate(zip(bars2, pcts2)):
        ax2.annotate(pct, xy=(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10, fontweight='bold', color='red')

    ax2.set_yticks(y2)
    ax2.set_yticklabels(names2, fontsize=10)
    ax2.set_xlabel('Holdout F1', fontsize=11)
    ax2.set_title("WHAT DIDN'T WORK", fontsize=14, fontweight='bold', color='red')
    ax2.set_xlim(0.63, 0.72)
    ax2.legend(loc='lower right')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_worked_vs_didnt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_worked_vs_didnt.png'}")


def create_why_explanation():
    """Create infographic explaining WHY each technique worked or didn't."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, 'WHY Did Each Technique Work (or Not)?', fontsize=22, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Feature Selection box
    fs_text = """
    FEATURE SELECTION (+3.7%)
    ========================
    WHY IT WORKED:
    • Reduced 28 features → 10 most predictive
    • Removed noise that caused overfitting
    • Key features: glucose, BMI, age bins
    • RFE found optimal subset systematically

    WHY IT'S #1:
    • Less features = less overfitting
    • Kept only signal, removed noise
    """
    ax.text(0.02, 0.75, fs_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # SMOTE box
    smote_text = """
    SMOTE (+1.8%)
    =============
    WHY IT WORKED:
    • Created synthetic minority samples
    • Helped model see more positive examples
    • k=5 neighbors was optimal balance

    WHY k=5 BEAT k=3, k=7:
    • k=3: too few neighbors = noisy samples
    • k=7: too many = samples too similar
    • k=5: goldilocks zone
    """
    ax.text(0.27, 0.75, smote_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Calibration box
    cal_text = """
    CALIBRATION (+1.5%)
    ===================
    WHY IT WORKED:
    • Sigmoid scaling fixed probability estimates
    • Lower threshold (0.4) caught more positives
    • Better probabilities → better F1

    WHY SIGMOID > ISOTONIC:
    • Isotonic needs more data to fit properly
    • 768 samples too few for isotonic
    • Sigmoid's 2 parameters more stable
    """
    ax.text(0.52, 0.75, cal_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Regularization box
    reg_text = """
    REGULARIZATION (+1.2%)
    ======================
    WHY IT WORKED:
    • C=0.5 was moderate, not extreme
    • L2 penalty smoothed coefficients
    • Reduced overfitting on small data

    WHY C=0.5 > C=0.001:
    • Too strong reg → underfitting
    • C=0.5 balances bias/variance
    • Dataset isn't THAT small (768)
    """
    ax.text(0.77, 0.75, reg_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Stacking box (didn't work)
    stack_text = """
    STACKING (-3.2%)
    ================
    WHY IT FAILED:
    • Added complexity without benefit
    • 4 models → more parameters to fit
    • CV-holdout gap was 0.084 (overfit!)

    THE LESSON:
    • More models ≠ better on small data
    • Simpler models generalize better
    • Complexity is the enemy here
    """
    ax.text(0.02, 0.35, stack_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # MLP box (didn't work)
    mlp_text = """
    MLP NEURAL NET (0%)
    ===================
    WHY IT TIED (NOT BEAT):
    • Neural nets need more data
    • 768 samples is tiny for deep learning
    • Same inductive bias as LogReg here

    THE LESSON:
    • Don't use neural nets on tiny data
    • Tabular data prefers trees/linear
    • No free lunch from architecture
    """
    ax.text(0.27, 0.35, mlp_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # CV Strategy box
    cv_text = """
    CV STRATEGY INSIGHT
    ===================
    FINDING:
    • 8+2 holdout was PESSIMISTIC
    • True performance ~0.690 not 0.665
    • Gap to target is 5.5% not 8%

    RECOMMENDATION:
    • Use 6+4 or repeated 5-fold
    • More holdout folds = stable estimate
    • 2 folds has too much variance
    """
    ax.text(0.52, 0.35, cv_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Summary box
    summary_text = """
    BOTTOM LINE
    ===========
    Best single technique: Feature Selection (+3.7%)
    Combined best:         0.690 holdout F1

    Still short of Auto-sklearn (0.745) by 5.5%

    WHY THE GAP REMAINS:
    • Auto-sklearn uses 100s of configs
    • Has meta-learning from prior tasks
    • Uses advanced stacking that works
    • More sophisticated HPO

    WE LEARNED:
    • Simpler > Complex on small data
    • Feature selection is underrated
    • SMOTE helps but modestly
    • Neural nets need more data
    """
    ax.text(0.77, 0.35, summary_text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(VISUALS_DIR / 'diabetes_why_explanation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_why_explanation.png'}")


def create_gap_to_target():
    """Show remaining gap to Auto-sklearn target."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Progress visualization
    baseline = 0.595
    current_best = 0.690
    target = 0.745

    # Calculate progress
    total_gap = target - baseline
    closed_gap = current_best - baseline
    remaining_gap = target - current_best
    progress_pct = (closed_gap / total_gap) * 100

    # Create stacked bar
    ax.barh(['Progress'], [closed_gap], left=[baseline], color='#2ecc71',
            edgecolor='black', linewidth=2, label=f'Closed ({closed_gap:.3f})')
    ax.barh(['Progress'], [remaining_gap], left=[current_best], color='#e74c3c',
            edgecolor='black', linewidth=2, label=f'Remaining ({remaining_gap:.3f})')

    # Add markers
    ax.axvline(x=baseline, color='blue', linestyle='-', linewidth=3, label=f'Baseline ({baseline})')
    ax.axvline(x=current_best, color='green', linestyle='-', linewidth=3, label=f'Current Best ({current_best})')
    ax.axvline(x=target, color='red', linestyle='-', linewidth=3, label=f'Target ({target})')

    # Add percentage labels
    ax.text((baseline + current_best) / 2, 0, f'{progress_pct:.0f}%\nClosed',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text((current_best + target) / 2, 0, f'{100-progress_pct:.0f}%\nRemaining',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    ax.set_xlim(0.55, 0.80)
    ax.set_xlabel('Holdout F1 Score', fontsize=12)
    ax.set_title(f'Progress Toward Auto-sklearn Target\n(Closed {progress_pct:.0f}% of the gap)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'diabetes_gap_to_target.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_gap_to_target.png'}")


def create_technique_summary_table():
    """Create a visual summary table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    headers = ['Technique', 'Best F1', 'vs Base', 'CV Gap', 'Verdict']
    data = [
        ['Feature Selection (RFE)', '0.690', '+3.7%', '0.016', 'BEST'],
        ['SMOTE (k=5)', '0.677', '+1.8%', '0.027', 'GOOD'],
        ['Calibration (Sigmoid)', '0.675', '+1.5%', '0.038', 'GOOD'],
        ['Regularization (C=0.5)', '0.673', '+1.2%', '0.020', 'GOOD'],
        ['Binning (4 quantile)', '0.666', '+0.2%', '0.047', 'MARGINAL'],
        ['MLP (128,64)', '0.665', '0%', '0.001', 'NO GAIN'],
        ['Stacking (4 models)', '0.644', '-3.2%', '0.084', 'WORSE'],
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.12, 0.15])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Color header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color rows by verdict
    verdict_colors = {
        'BEST': '#2ecc71',
        'GOOD': '#a8e6cf',
        'MARGINAL': '#ffeaa7',
        'NO GAIN': '#fab1a0',
        'WORSE': '#e74c3c'
    }

    for i, row in enumerate(data):
        verdict = row[-1]
        color = verdict_colors.get(verdict, 'white')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            if verdict == 'WORSE':
                table[(i+1, j)].set_text_props(color='white')

    ax.set_title('Diabetes Improvement Experiments: Summary Table\n',
                 fontsize=16, fontweight='bold')

    plt.savefig(VISUALS_DIR / 'diabetes_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {VISUALS_DIR / 'diabetes_summary_table.png'}")


if __name__ == '__main__':
    print("Creating comprehensive experiment visualizations...")
    create_master_comparison()
    create_improvement_waterfall()
    create_what_worked_didnt()
    create_why_explanation()
    create_gap_to_target()
    create_technique_summary_table()
    print("\nAll visualizations created!")
