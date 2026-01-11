"""
Visualization Generator for Chess Evolution

Generates SVG charts to document evolution progress:
1. Fitness bar chart (per generation)
2. Evolution trajectory (cumulative)
3. Strategy comparison heatmap
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CandidateResult:
    """Result for a single candidate"""
    id: str
    generation: int
    acpl: float
    legal_rate: float
    best_rate: float
    is_champion: bool
    is_valid: bool
    strategy: str


def generate_fitness_bar_chart(
    candidates: List[CandidateResult],
    generation: int,
    output_path: str
) -> str:
    """Generate SVG bar chart of candidate fitness for one generation"""

    # Sort by ACPL (lower is better)
    sorted_candidates = sorted(candidates, key=lambda c: c.acpl if c.is_valid else 9999)

    # SVG dimensions
    width = 800
    height = 100 + len(sorted_candidates) * 50
    bar_max_width = 500
    margin_left = 200
    margin_top = 60
    bar_height = 35
    bar_spacing = 50

    # Find max ACPL for scaling (cap at 1000)
    max_acpl = min(1000, max(c.acpl for c in sorted_candidates if c.is_valid))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '  <style>',
        '    .title { font: bold 20px sans-serif; fill: #333; }',
        '    .label { font: 14px monospace; fill: #333; }',
        '    .value { font: bold 14px sans-serif; fill: #333; }',
        '    .bar-good { fill: #4CAF50; }',
        '    .bar-medium { fill: #FFC107; }',
        '    .bar-bad { fill: #F44336; }',
        '    .bar-invalid { fill: #9E9E9E; }',
        '    .champion { fill: #FFD700; }',
        '    .axis { stroke: #666; stroke-width: 1; }',
        '  </style>',
        f'  <text x="{width//2}" y="30" text-anchor="middle" class="title">Generation {generation} Fitness (ACPL - Lower is Better)</text>',
    ]

    # Draw bars
    for i, candidate in enumerate(sorted_candidates):
        y = margin_top + i * bar_spacing

        # Calculate bar width
        if candidate.is_valid:
            bar_width = (candidate.acpl / max_acpl) * bar_max_width
            bar_width = min(bar_width, bar_max_width)

            # Color based on ACPL
            if candidate.acpl < 100:
                bar_class = "bar-good"
            elif candidate.acpl < 300:
                bar_class = "bar-medium"
            else:
                bar_class = "bar-bad"
        else:
            bar_width = bar_max_width
            bar_class = "bar-invalid"

        # Champion marker
        champion_marker = " ★" if candidate.is_champion else ""
        invalid_marker = " ✗" if not candidate.is_valid else ""

        # Label
        svg_lines.append(f'  <text x="{margin_left - 10}" y="{y + bar_height//2 + 5}" text-anchor="end" class="label">{candidate.id}</text>')

        # Bar
        svg_lines.append(f'  <rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" class="{bar_class}" rx="3"/>')

        # Value
        acpl_str = f"{candidate.acpl:.1f}" if candidate.is_valid else "INVALID"
        svg_lines.append(f'  <text x="{margin_left + bar_width + 10}" y="{y + bar_height//2 + 5}" class="value">{acpl_str} ACPL{champion_marker}{invalid_marker}</text>')

    # Legend
    legend_y = height - 30
    svg_lines.extend([
        f'  <rect x="50" y="{legend_y}" width="15" height="15" class="bar-good"/>',
        f'  <text x="70" y="{legend_y + 12}" class="label">&lt;100</text>',
        f'  <rect x="130" y="{legend_y}" width="15" height="15" class="bar-medium"/>',
        f'  <text x="150" y="{legend_y + 12}" class="label">100-300</text>',
        f'  <rect x="220" y="{legend_y}" width="15" height="15" class="bar-bad"/>',
        f'  <text x="240" y="{legend_y + 12}" class="label">&gt;300</text>',
        f'  <text x="320" y="{legend_y + 12}" class="label">★ Champion</text>',
    ])

    svg_lines.append('</svg>')

    svg_content = '\n'.join(svg_lines)

    with open(output_path, 'w') as f:
        f.write(svg_content)

    return svg_content


def generate_trajectory_chart(
    history: List[Dict[str, Any]],
    output_path: str
) -> str:
    """Generate SVG line chart of ACPL evolution over generations"""

    if not history:
        return ""

    width = 800
    height = 400
    margin = {"top": 50, "right": 50, "bottom": 50, "left": 80}
    chart_width = width - margin["left"] - margin["right"]
    chart_height = height - margin["top"] - margin["bottom"]

    # Extract data
    generations = [h["generation"] for h in history]
    best_acpls = [h["best_acpl"] for h in history]

    # Scale
    max_gen = max(generations) if generations else 1
    max_acpl = max(best_acpls) if best_acpls else 1000

    def scale_x(gen):
        return margin["left"] + (gen / max(max_gen, 1)) * chart_width

    def scale_y(acpl):
        return margin["top"] + chart_height - (acpl / max_acpl) * chart_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '  <style>',
        '    .title { font: bold 18px sans-serif; fill: #333; }',
        '    .axis-label { font: 12px sans-serif; fill: #666; }',
        '    .axis { stroke: #666; stroke-width: 1; }',
        '    .grid { stroke: #eee; stroke-width: 1; }',
        '    .line { fill: none; stroke: #2196F3; stroke-width: 3; }',
        '    .point { fill: #2196F3; }',
        '    .point-label { font: 10px sans-serif; fill: #333; }',
        '  </style>',
        f'  <text x="{width//2}" y="25" text-anchor="middle" class="title">Evolution Trajectory - Best ACPL Over Generations</text>',
    ]

    # Y-axis
    svg_lines.append(f'  <line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" class="axis"/>')

    # X-axis
    svg_lines.append(f'  <line x1="{margin["left"]}" y1="{height - margin["bottom"]}" x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" class="axis"/>')

    # Y-axis labels
    for acpl in [0, max_acpl * 0.25, max_acpl * 0.5, max_acpl * 0.75, max_acpl]:
        y = scale_y(acpl)
        svg_lines.append(f'  <text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" class="axis-label">{int(acpl)}</text>')
        svg_lines.append(f'  <line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid"/>')

    # X-axis labels
    for gen in range(0, max_gen + 1):
        x = scale_x(gen)
        svg_lines.append(f'  <text x="{x}" y="{height - margin["bottom"] + 20}" text-anchor="middle" class="axis-label">{gen}</text>')

    # Axis titles
    svg_lines.append(f'  <text x="{width//2}" y="{height - 10}" text-anchor="middle" class="axis-label">Generation</text>')
    svg_lines.append(f'  <text x="15" y="{height//2}" text-anchor="middle" transform="rotate(-90, 15, {height//2})" class="axis-label">ACPL</text>')

    # Line path
    if len(generations) > 1:
        points = " ".join([f"{scale_x(g)},{scale_y(a)}" for g, a in zip(generations, best_acpls)])
        svg_lines.append(f'  <polyline points="{points}" class="line"/>')

    # Points
    for gen, acpl in zip(generations, best_acpls):
        x, y = scale_x(gen), scale_y(acpl)
        svg_lines.append(f'  <circle cx="{x}" cy="{y}" r="5" class="point"/>')
        svg_lines.append(f'  <text x="{x}" y="{y - 10}" text-anchor="middle" class="point-label">{int(acpl)}</text>')

    svg_lines.append('</svg>')

    svg_content = '\n'.join(svg_lines)

    with open(output_path, 'w') as f:
        f.write(svg_content)

    return svg_content


def generate_results_markdown(
    generation: int,
    candidates: List[CandidateResult],
    history: List[Dict[str, Any]],
    learnings: Dict[str, List[str]],
    output_path: str
) -> str:
    """Generate GEN{N}_RESULTS.md documentation"""

    sorted_candidates = sorted(candidates, key=lambda c: c.acpl if c.is_valid else 9999)
    champion = next((c for c in sorted_candidates if c.is_champion), None)

    prev_best = history[-2]["best_acpl"] if len(history) >= 2 else None
    curr_best = history[-1]["best_acpl"] if history else None

    improvement = ""
    if prev_best and curr_best:
        pct_change = ((prev_best - curr_best) / prev_best) * 100
        improvement = f" ({pct_change:+.1f}%)"

    md_lines = [
        f"# Generation {generation} Results",
        "",
        "## Summary",
        "",
        f"- **Best ACPL**: {curr_best:.1f}{improvement}" if curr_best else "- **Best ACPL**: N/A",
        f"- **Champion**: {champion.id if champion else 'N/A'}",
        f"- **Valid Candidates**: {sum(1 for c in candidates if c.is_valid)}/{len(candidates)}",
        "",
        "## Fitness Rankings",
        "",
        "| Rank | Candidate | ACPL | Legal Rate | Best Move Rate | Status |",
        "|------|-----------|------|------------|----------------|--------|",
    ]

    for i, c in enumerate(sorted_candidates, 1):
        status = "★ Champion" if c.is_champion else ("✓ Valid" if c.is_valid else "✗ Invalid")
        acpl_str = f"{c.acpl:.1f}" if c.is_valid else "N/A"
        legal_str = f"{c.legal_rate*100:.1f}%" if c.is_valid else "N/A"
        best_str = f"{c.best_rate*100:.1f}%" if c.is_valid else "N/A"
        md_lines.append(f"| {i} | {c.id} | {acpl_str} | {legal_str} | {best_str} | {status} |")

    md_lines.extend([
        "",
        "## Visualizations",
        "",
        f"![Fitness Chart](../visualizations/fitness_gen{generation}.svg)",
        "",
        "![Evolution Trajectory](../visualizations/evolution_trajectory.svg)",
        "",
        "## What Worked",
        "",
    ])

    for item in learnings.get("worked", ["- TBD"]):
        md_lines.append(f"- {item}")

    md_lines.extend([
        "",
        "## What Didn't Work",
        "",
    ])

    for item in learnings.get("failed", ["- TBD"]):
        md_lines.append(f"- {item}")

    md_lines.extend([
        "",
        "## Key Learnings",
        "",
    ])

    for item in learnings.get("learnings", ["- TBD"]):
        md_lines.append(f"- {item}")

    md_lines.extend([
        "",
        "## Next Generation Strategy",
        "",
    ])

    for item in learnings.get("next_strategy", ["- Continue evolution with mutations of champion"]):
        md_lines.append(f"- {item}")

    md_content = '\n'.join(md_lines)

    with open(output_path, 'w') as f:
        f.write(md_content)

    return md_content


if __name__ == "__main__":
    # Demo with sample data
    print("Generating sample visualizations...")

    # Sample candidates
    candidates = [
        CandidateResult("gen0_baseline", 0, 150.5, 0.98, 0.12, False, True, "balanced"),
        CandidateResult("gen0_elite_only", 0, 89.3, 0.99, 0.18, True, True, "quality"),
        CandidateResult("gen0_endgame", 0, 125.7, 0.97, 0.15, False, True, "endgame"),
        CandidateResult("gen0_tactical", 0, 110.2, 0.95, 0.14, False, True, "tactical"),
        CandidateResult("gen0_failed", 0, 0, 0.5, 0, False, False, "failed"),
    ]

    history = [
        {"generation": 0, "best_acpl": 89.3, "best_id": "gen0_elite_only"},
    ]

    learnings = {
        "worked": ["High Elo filtering (2200+) improved move quality"],
        "failed": ["Curriculum learning showed no improvement"],
        "learnings": ["Quality > quantity for training data"],
        "next_strategy": ["Mutate elite_only config with different prompt formats"],
    }

    # Create output dirs
    Path("visualizations").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Generate visualizations
    generate_fitness_bar_chart(candidates, 0, "visualizations/fitness_gen0.svg")
    generate_trajectory_chart(history, "visualizations/evolution_trajectory.svg")
    generate_results_markdown(0, candidates, history, learnings, "results/GEN0_RESULTS.md")

    print("Generated:")
    print("  - visualizations/fitness_gen0.svg")
    print("  - visualizations/evolution_trajectory.svg")
    print("  - results/GEN0_RESULTS.md")
