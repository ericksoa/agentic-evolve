#!/usr/bin/env python3
"""
Visualize the key findings from winner analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

# Data from analysis
datasets = {
    'credit-g': {'gain': 16.8, 'overlap': 85.3, 'separation': 0.062, 'best_t': 0.10},
    'mozilla4': {'gain': 8.2, 'overlap': 36.8, 'separation': 0.391, 'best_t': 0.33},
    'diabetes': {'gain': 1.7, 'overlap': 37.8, 'separation': 0.325, 'best_t': 0.43},
    'pc1': {'gain': 1.7, 'overlap': 39.0, 'separation': 0.310, 'best_t': 0.51},
    'phoneme': {'gain': 0.4, 'overlap': 34.8, 'separation': 0.303, 'best_t': 0.54},
    'spambase': {'gain': 0.0, 'overlap': 10.9, 'separation': 0.736, 'best_t': 0.49},
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

names = list(datasets.keys())
gains = [datasets[n]['gain'] for n in names]
overlaps = [datasets[n]['overlap'] for n in names]
separations = [datasets[n]['separation'] for n in names]
thresholds = [datasets[n]['best_t'] for n in names]

colors = ['#2ecc71' if g > 5 else '#e74c3c' if g < 1 else '#f39c12' for g in gains]

# Plot 1: Overlap vs Gain
ax1 = axes[0]
ax1.scatter(overlaps, gains, c=colors, s=100)
for i, name in enumerate(names):
    ax1.annotate(name, (overlaps[i], gains[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax1.set_xlabel('Overlap Zone (%)', fontsize=11)
ax1.set_ylabel('Threshold Optimization Gain (%)', fontsize=11)
ax1.set_title('More Uncertainty = More Gain', fontsize=12, fontweight='bold')
ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax1.text(60, 5.5, 'Big Win Zone', fontsize=9, color='gray')

# Add correlation line
z = np.polyfit(overlaps, gains, 1)
p = np.poly1d(z)
x_line = np.linspace(min(overlaps), max(overlaps), 100)
ax1.plot(x_line, p(x_line), 'b--', alpha=0.3, label=f'r=+0.88')
ax1.legend(loc='upper left')

# Plot 2: Class Separation vs Gain
ax2 = axes[1]
ax2.scatter(separations, gains, c=colors, s=100)
for i, name in enumerate(names):
    ax2.annotate(name, (separations[i], gains[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('Class Separation (prob difference)', fontsize=11)
ax2.set_ylabel('Threshold Optimization Gain (%)', fontsize=11)
ax2.set_title('Less Separation = More Gain', fontsize=12, fontweight='bold')
ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)

# Add correlation line
z = np.polyfit(separations, gains, 1)
p = np.poly1d(z)
x_line = np.linspace(min(separations), max(separations), 100)
ax2.plot(x_line, p(x_line), 'b--', alpha=0.3, label=f'r=-0.66')
ax2.legend(loc='upper right')

# Plot 3: Optimal Threshold by dataset
ax3 = axes[2]
bars = ax3.barh(names, thresholds, color=colors)
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default (0.5)')
ax3.set_xlabel('Optimal Threshold', fontsize=11)
ax3.set_title('Winners Need Lower Thresholds', fontsize=12, fontweight='bold')
ax3.legend()
ax3.set_xlim(0, 0.7)

# Add threshold values on bars
for i, (bar, t) in enumerate(zip(bars, thresholds)):
    ax3.text(t + 0.02, bar.get_y() + bar.get_height()/2, f'{t:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: threshold_analysis.png")

# Print summary
print("\n" + "="*60)
print("KEY FINDING: Threshold optimization helps when model is UNCERTAIN")
print("="*60)
print("""
The Pattern:
┌─────────────────────────────────────────────────────────────┐
│  HIGH overlap (>40%)  →  Model uncertain  →  BIG GAINS      │
│  LOW overlap (<20%)   →  Model confident  →  NO GAINS       │
└─────────────────────────────────────────────────────────────┘

Why credit-g wins big (+16.8%):
• 85% of samples in the "uncertain zone" (probs 0.3-0.7)
• Model can barely separate classes (0.06 prob difference)
• Default 0.5 threshold is TERRIBLE for this distribution
• Optimal threshold: 0.10 (!!!)

Why spambase sees no gain:
• Only 11% in uncertain zone
• Model is confident and correct (0.74 separation)
• 0.5 threshold is already near-optimal

ACTIONABLE INSIGHT:
→ Detect overlap zone before choosing strategy
→ High overlap (>40%) = use aggressive threshold search
→ Low overlap (<20%) = skip threshold optimization
""")
