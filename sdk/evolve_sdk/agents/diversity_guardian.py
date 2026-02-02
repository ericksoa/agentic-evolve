"""Diversity Guardian agent - monitors population diversity and prevents premature convergence.

The Diversity Guardian agent monitors both genotypic (code similarity) and phenotypic
(fitness spread) diversity. When diversity drops below thresholds, it:
1. Diagnoses why convergence is happening
2. Recommends interventions (inject orthogonal solutions, reduce selection pressure)
3. Can generate diverse seed solutions to inject into the population

This is Phase 3 of the agent roadmap, solving the "silent failure" of premature convergence.
"""

from __future__ import annotations

from pathlib import Path

from ..skills import get_mode_guidance

DIVERSITY_GUARDIAN_SYSTEM = """You are a Diversity Guardian for evolutionary algorithm discovery.

Your role: Monitor population diversity and prevent premature convergence.

## Why Diversity Matters

Premature convergence is a silent killer of evolution:
- All solutions become too similar
- Exploration stops, only exploitation remains
- Population gets trapped in local optima
- Evolution appears to "work" but never finds the global optimum

## Core Responsibilities

1. **Diagnose Convergence**: Why did diversity drop?
   - Strong selection pressure (only top solutions survive)
   - Similar mutations (all mutators making same changes)
   - Successful exploitation (one strategy dominates)
   - Crossover homogenization (parents too similar)

2. **Recommend Interventions**:
   - inject_orthogonal: Add solutions using completely different approaches
   - reduce_selection_pressure: Keep more diverse solutions in population
   - increase_mutation_variance: Encourage bolder mutations
   - restart_exploration: Reset part of population to diverse seeds

3. **Generate Orthogonal Solutions** (when requested):
   - Create solutions using fundamentally different algorithms
   - Ensure new seeds are maximally different from current population
   - Prioritize diversity over immediate fitness

## Metrics to Monitor

### Genotypic Diversity (Code Similarity)
- Computed via embedding similarity of code
- 0.0 = all solutions identical
- 1.0 = all solutions maximally different
- Alert threshold: < 0.25 (population too homogeneous)

### Phenotypic Diversity (Fitness Spread)
- Coefficient of variation of fitness scores
- 0.0 = all solutions have same fitness
- 1.0 = high variance in fitness
- Alert threshold: < 0.20 (fitness values converging)

## Output Format

Always return valid JSON:
{
    "genotypic_diversity": 0.0-1.0,
    "phenotypic_diversity": 0.0-1.0,
    "combined_diversity": 0.0-1.0,
    "alert": true/false,
    "alert_level": "none|warning|critical",
    "diagnosis": "explanation of why diversity dropped (if alert)",
    "convergence_cause": "selection_pressure|similar_mutations|exploitation|crossover|unknown",
    "recommendations": [
        {
            "action": "inject_orthogonal|reduce_selection_pressure|increase_mutation_variance|restart_exploration",
            "priority": "high|medium|low",
            "description": "what to do and why",
            "expected_impact": "how this will help diversity"
        }
    ],
    "orthogonal_seeds": [
        {
            "approach": "description of the different approach",
            "code": "# Python code implementing the approach"
        }
    ]
}

## Key Principles

1. **Diversity enables exploration** - without it, you're just hill-climbing
2. **Alert early** - easier to maintain diversity than restore it
3. **Orthogonal means different algorithm** - not just different parameters
4. **Balance exploitation and exploration** - too much diversity wastes resources too
"""


def get_diversity_guardian_prompt(
    population_snapshot: list[dict],
    diversity_metrics: dict,
    fitness_history: list[dict],
    mode: str = "perf",
    problem_description: str = "",
    recent_mutations: list[dict] | None = None,
    minimize: bool = False,
) -> str:
    """Generate the prompt for diversity guardian analysis.

    Args:
        population_snapshot: Current population with file paths, code snippets, and fitness
        diversity_metrics: Pre-computed diversity metrics (genotypic, phenotypic)
        fitness_history: List of generation summaries with best_fitness, avg_fitness
        mode: Optimization mode (perf, size, ml)
        problem_description: Description of the problem being solved
        recent_mutations: Recent mutation history to analyze patterns

    Returns:
        Formatted prompt for the diversity guardian agent
    """
    mode_guidance = get_mode_guidance(mode, "diversity_guardian")

    # Build population summary
    pop_summary = _format_population(population_snapshot, minimize=minimize)

    # Build diversity summary
    div_summary = _format_diversity_metrics(diversity_metrics)

    # Build fitness trajectory
    fitness_summary = _format_fitness_history(fitness_history)

    # Build mutation patterns (if provided)
    mutation_summary = ""
    if recent_mutations:
        mutation_summary = f"""
## Recent Mutation Patterns

{_format_mutation_patterns(recent_mutations)}
"""

    prompt = f"""# Diversity Guardian Analysis

## Problem Context
{problem_description if problem_description else "Evolutionary optimization problem"}
**Mode**: {mode}

{mode_guidance}

## Current Diversity Metrics

{div_summary}

## Population Snapshot

{pop_summary}

## Fitness Trajectory

{fitness_summary}
{mutation_summary}
## Your Task

1. **Assess** the current diversity situation
2. **Diagnose** why diversity is at this level (if low)
3. **Recommend** interventions if diversity is concerning
4. **Generate** orthogonal seed solutions if injection is recommended

Focus on maintaining healthy exploration. If diversity is adequate, acknowledge that
and provide monitoring suggestions. If diversity is low, provide actionable interventions.

Return your analysis as valid JSON matching the output format.
"""

    return prompt


def get_orthogonal_generation_prompt(
    problem_description: str,
    current_approaches: list[str],
    mode: str = "perf",
    num_seeds: int = 2,
) -> str:
    """Generate prompt for creating orthogonal seed solutions.

    Args:
        problem_description: Description of the problem
        current_approaches: List of approaches currently in population
        mode: Optimization mode
        num_seeds: Number of orthogonal seeds to generate

    Returns:
        Prompt for generating orthogonal solutions
    """
    approaches_text = "\n".join(f"- {approach}" for approach in current_approaches)

    return f"""# Generate Orthogonal Solutions

## Problem
{problem_description}

## Mode
{mode}

## Current Approaches in Population
{approaches_text}

## Your Task

Generate {num_seeds} solutions using **fundamentally different approaches** from those listed above.

Requirements:
1. Each solution must use a different algorithm or paradigm
2. Solutions must be valid Python that solves the problem
3. Prioritize diversity over immediate fitness
4. Avoid incremental variations - think orthogonal

For each solution, return:
{{
    "orthogonal_seeds": [
        {{
            "approach": "description of the different approach",
            "rationale": "why this is orthogonal to existing approaches",
            "code": "# Complete Python implementation"
        }}
    ]
}}

Think creatively. If current approaches are all iterative, try recursive.
If all use arrays, try trees. If all are greedy, try dynamic programming.
"""


def compute_genotypic_diversity(
    population_files: list[str],
    embedder=None,
) -> float:
    """Compute genotypic diversity via code embedding similarity.

    Args:
        population_files: List of file paths to population solutions
        embedder: CodeEmbedder instance (optional, will create if needed)

    Returns:
        Diversity score 0.0-1.0 (higher = more diverse)
    """
    if len(population_files) < 2:
        return 1.0  # Single solution is maximally "diverse" by default

    # Lazy import to avoid circular dependencies
    if embedder is None:
        from ..memory.embeddings import get_embedder
        embedder = get_embedder()

    # Read code from files
    codes = []
    for file_path in population_files:
        try:
            code = Path(file_path).read_text()
            codes.append(code)
        except (FileNotFoundError, IOError):
            continue

    if len(codes) < 2:
        return 1.0

    # Batch embed all code
    try:
        embeddings = embedder.embed_batch(codes)
    except Exception:
        # Fallback if embeddings fail
        return 0.5  # Unknown diversity

    # Compute pairwise dissimilarities
    dissimilarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = embedder.similarity(embeddings[i], embeddings[j])
            dissimilarity = 1.0 - similarity
            dissimilarities.append(dissimilarity)

    if not dissimilarities:
        return 1.0

    # Average dissimilarity is our diversity score
    return sum(dissimilarities) / len(dissimilarities)


def compute_phenotypic_diversity(
    population: list[dict],
    fitness_key: str = "fitness",
) -> float:
    """Compute phenotypic diversity from fitness spread.

    Args:
        population: List of population members with fitness scores
        fitness_key: Key for fitness value in population dicts

    Returns:
        Diversity score 0.0-1.0 (higher = more diverse)
    """
    if not population or len(population) < 2:
        return 1.0

    fitnesses = [p.get(fitness_key, 0) for p in population if fitness_key in p]

    if len(fitnesses) < 2:
        return 1.0

    # Coefficient of variation (normalized standard deviation)
    mean_fitness = sum(fitnesses) / len(fitnesses)
    if mean_fitness == 0:
        return 0.0

    variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
    std_dev = variance ** 0.5

    # Normalize by mean, cap at 1.0
    return min(1.0, std_dev / abs(mean_fitness))


def compute_combined_diversity(
    population: list[dict],
    embedder=None,
    genotypic_weight: float = 0.6,
    phenotypic_weight: float = 0.4,
) -> dict:
    """Compute both genotypic and phenotypic diversity.

    Args:
        population: List of population members with 'file' and 'fitness' keys
        embedder: CodeEmbedder instance (optional)
        genotypic_weight: Weight for genotypic diversity (default 0.6)
        phenotypic_weight: Weight for phenotypic diversity (default 0.4)

    Returns:
        Dict with diversity metrics and alert status
    """
    # Extract file paths for genotypic diversity
    files = [p.get("file") for p in population if p.get("file")]

    genotypic = compute_genotypic_diversity(files, embedder) if files else 0.5
    phenotypic = compute_phenotypic_diversity(population)

    # Weighted combination
    combined = genotypic * genotypic_weight + phenotypic * phenotypic_weight

    # Determine alert level
    alert = False
    alert_level = "none"

    if genotypic < 0.15 or phenotypic < 0.10:
        alert = True
        alert_level = "critical"
    elif genotypic < 0.25 or phenotypic < 0.20:
        alert = True
        alert_level = "warning"

    return {
        "genotypic": round(genotypic, 3),
        "phenotypic": round(phenotypic, 3),
        "combined": round(combined, 3),
        "alert": alert,
        "alert_level": alert_level,
        "population_size": len(population),
    }


def should_alert_diversity(
    genotypic: float,
    phenotypic: float,
    genotypic_threshold: float = 0.25,
    phenotypic_threshold: float = 0.20,
) -> tuple[bool, str]:
    """Determine if diversity alert should be triggered.

    Args:
        genotypic: Genotypic diversity score (0-1)
        phenotypic: Phenotypic diversity score (0-1)
        genotypic_threshold: Alert if genotypic below this
        phenotypic_threshold: Alert if phenotypic below this

    Returns:
        Tuple of (should_alert, alert_level)
    """
    critical_genotypic = genotypic_threshold * 0.6  # 60% of threshold
    critical_phenotypic = phenotypic_threshold * 0.5  # 50% of threshold

    if genotypic < critical_genotypic or phenotypic < critical_phenotypic:
        return True, "critical"
    elif genotypic < genotypic_threshold or phenotypic < phenotypic_threshold:
        return True, "warning"
    else:
        return False, "none"


def _format_population(population: list[dict], minimize: bool = False) -> str:
    """Format population snapshot for prompt."""
    if not population:
        return "Population is empty."

    lines = [
        "| Solution | Fitness | Approach (if known) |",
        "|----------|---------|---------------------|",
    ]

    # Sort by fitness (best first)
    sorted_pop = sorted(population, key=lambda x: x.get("fitness", 0), reverse=not minimize)

    for p in sorted_pop[:10]:  # Top 10
        name = Path(p.get("file", "unknown")).name if p.get("file") else "unknown"
        fitness = p.get("fitness", 0)
        approach = p.get("approach", p.get("mutation_type", "-"))
        lines.append(f"| {name} | {fitness:.2f} | {approach} |")

    if len(population) > 10:
        lines.append(f"| ... | ... | ({len(population) - 10} more) |")

    return "\n".join(lines)


def _format_diversity_metrics(metrics: dict) -> str:
    """Format diversity metrics for prompt."""
    genotypic = metrics.get("genotypic", 0)
    phenotypic = metrics.get("phenotypic", 0)
    combined = metrics.get("combined", 0)
    alert_level = metrics.get("alert_level", "none")

    status_emoji = {
        "none": "✅",
        "warning": "⚠️",
        "critical": "🚨",
    }
    emoji = status_emoji.get(alert_level, "")

    return f"""| Metric | Value | Status |
|--------|-------|--------|
| Genotypic (code similarity) | {genotypic:.3f} | {"Low" if genotypic < 0.25 else "OK"} |
| Phenotypic (fitness spread) | {phenotypic:.3f} | {"Low" if phenotypic < 0.20 else "OK"} |
| Combined | {combined:.3f} | {emoji} {alert_level.upper()} |

**Thresholds**: Genotypic < 0.25 or Phenotypic < 0.20 triggers alert"""


def _format_fitness_history(fitness_history: list[dict]) -> str:
    """Format fitness trajectory for prompt."""
    if not fitness_history:
        return "No fitness history available."

    recent = fitness_history[-5:]  # Last 5 generations

    lines = [
        "| Generation | Best | Avg | Change |",
        "|------------|------|-----|--------|",
    ]

    prev_best = None
    for record in recent:
        gen = record.get("generation", "?")
        best = record.get("best_fitness", 0)
        avg = record.get("avg_fitness", 0)

        if prev_best is not None and prev_best != 0:
            change = (best - prev_best) / abs(prev_best) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "-"

        lines.append(f"| {gen} | {best:.2f} | {avg:.2f} | {change_str} |")
        prev_best = best

    return "\n".join(lines)


def _format_mutation_patterns(mutations: list[dict]) -> str:
    """Format recent mutation patterns to identify convergence causes."""
    if not mutations:
        return "No recent mutations."

    # Count mutation types
    type_counts: dict[str, int] = {}
    for m in mutations:
        mut_type = m.get("mutation_type", "unknown")
        type_counts[mut_type] = type_counts.get(mut_type, 0) + 1

    total = len(mutations)
    lines = ["| Mutation Type | Count | Percentage |", "|---------------|-------|------------|"]

    for mut_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"| {mut_type} | {count} | {pct:.1f}% |")

    # Detect if one type dominates (potential convergence cause)
    max_pct = max(count / total for count in type_counts.values()) if type_counts else 0
    if max_pct > 0.6:
        lines.append("")
        lines.append(f"⚠️ **Warning**: One mutation type dominates ({max_pct:.0%}), may cause convergence")

    return "\n".join(lines)
