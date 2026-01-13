"""Meta-Strategist agent - analyzes evolution progress and adjusts strategy.

The Meta-Strategist agent is triggered every N generations (default: 5).
Its role is to:
1. Analyze mutation success rates by type
2. Compare crossover vs mutation effectiveness
3. Track population diversity
4. Adjust strategy mix based on what's working

This makes evolution smarter by learning from its own history.
"""

from __future__ import annotations

from ..skills import get_mode_guidance

META_STRATEGIST_SYSTEM = """You are a meta-strategist for evolutionary algorithm discovery.

Your role: Analyze evolution progress and recommend strategy adjustments.

## Core Responsibilities

1. **Mutation Analysis**: Calculate success rates by mutation type
2. **Crossover Evaluation**: Measure contribution of crossover to fitness gains
3. **Diversity Tracking**: Assess population diversity (genotypic and phenotypic)
4. **Strategy Adjustment**: Recommend changes to mutation mix and parameters
5. **Trend Detection**: Identify patterns in what's working and what's not

## Key Metrics

### Mutation Effectiveness
For each mutation type (parameter_tweak, structural, algorithm_swap, etc.):
- Success rate = valid mutations / attempted mutations
- Impact = average fitness delta when successful
- Effectiveness = success_rate * impact

### Crossover Contribution
- Track whether crossover children (genXx) outperform their parents
- Contribution = (crossover_wins / total_crossover_attempts)

### Diversity Index
- Genotypic: How different is the code between solutions (0-1)
- Phenotypic: Spread of fitness values in population (0-1)
- Low diversity (<0.2) = premature convergence risk

## Strategy Adjustments

You can recommend:
- **increase_<type>_mutations**: Allocate more attempts to effective strategies
- **decrease_<type>_mutations**: Reduce attempts for low-performing strategies
- **adjust_crossover_frequency**: Increase/decrease crossover rate
- **inject_diversity**: Request orthogonal solutions when diversity is low
- **reset_exploration**: Suggest broader search when stuck in local region

## Output Format

Always return valid JSON:
{
    "generation_analyzed": <current generation>,
    "analysis_window": <how many generations analyzed>,
    "analysis": {
        "mutation_effectiveness": {
            "<type>": {"attempts": N, "successes": N, "avg_impact": X, "effectiveness": X},
            ...
        },
        "crossover_contribution": {
            "attempts": N,
            "wins": N,
            "contribution_rate": X
        },
        "diversity": {
            "genotypic": X,
            "phenotypic": X,
            "trend": "increasing|stable|decreasing"
        },
        "fitness_trajectory": {
            "current_best": X,
            "improvement_rate": X,
            "trend": "accelerating|stable|decelerating|stalled"
        }
    },
    "recommendations": [
        {
            "action": "<action_type>",
            "rationale": "<why this change>",
            "priority": "high|medium|low",
            "expected_impact": "<what we expect to happen>"
        }
    ],
    "strategy_update": {
        "mutation_weights": {"<type>": weight, ...},
        "crossover_rate": X,
        "exploration_vs_exploitation": X
    }
}

Be data-driven. Base recommendations on actual metrics, not hunches.
"""


def get_meta_strategist_prompt(
    generation: int,
    mutation_history: list[dict],
    crossover_history: list[dict],
    fitness_history: list[dict],
    population_snapshot: list[dict] | None = None,
    mode: str = "perf",
    analysis_window: int = 5,
) -> str:
    """Generate the prompt for meta-strategist analysis.

    Args:
        generation: Current generation number
        mutation_history: List of mutation records with type, success, fitness_delta
        crossover_history: List of crossover records with parent_fitnesses, child_fitness
        fitness_history: List of generation summaries with best_fitness, avg_fitness
        population_snapshot: Current population with file paths and fitness scores
        mode: Optimization mode (perf, size, ml)
        analysis_window: How many generations to analyze

    Returns:
        Formatted prompt for the meta-strategist agent
    """
    mode_guidance = get_mode_guidance(mode, "meta_strategist")

    # Filter to analysis window
    recent_mutations = [m for m in mutation_history if m.get("generation", 0) >= generation - analysis_window]
    recent_crossovers = [c for c in crossover_history if c.get("generation", 0) >= generation - analysis_window]
    recent_fitness = fitness_history[-(analysis_window + 1):] if len(fitness_history) > analysis_window else fitness_history

    # Build mutation summary
    mutation_summary = _summarize_mutations(recent_mutations)
    crossover_summary = _summarize_crossovers(recent_crossovers)
    fitness_summary = _summarize_fitness(recent_fitness)

    prompt = f"""# Meta-Strategy Analysis Request

## Context
- **Current Generation**: {generation}
- **Analysis Window**: Last {analysis_window} generations
- **Mode**: {mode}

{mode_guidance}

## Mutation History (last {analysis_window} generations)

{mutation_summary}

## Crossover History

{crossover_summary}

## Fitness Trajectory

{fitness_summary}
"""

    if population_snapshot:
        pop_summary = _summarize_population(population_snapshot)
        prompt += f"""
## Current Population

{pop_summary}
"""

    prompt += """
## Your Task

Analyze the evolution progress and provide:
1. **Effectiveness metrics** for each mutation type
2. **Crossover contribution** assessment
3. **Diversity analysis** (if population snapshot provided)
4. **Concrete recommendations** for strategy adjustments
5. **Updated weights** for mutation types and crossover rate

Focus on actionable insights. If something is working, do more of it.
If something isn't working, explain why and suggest alternatives.

Return your analysis as valid JSON matching the output format.
"""

    return prompt


def get_strategy_application_prompt(
    recommendations: list[dict],
    current_weights: dict,
    mode: str = "perf",
) -> str:
    """Generate prompt for applying strategy recommendations.

    Args:
        recommendations: List of recommendations from meta-strategist
        current_weights: Current mutation type weights
        mode: Optimization mode

    Returns:
        Prompt for applying the strategy changes
    """
    rec_text = "\n".join([
        f"- **{r['action']}** ({r.get('priority', 'medium')}): {r['rationale']}"
        for r in recommendations
    ])

    weights_text = "\n".join([f"- {k}: {v}" for k, v in current_weights.items()])

    return f"""# Apply Strategy Recommendations

## Current Mutation Weights
{weights_text}

## Recommendations to Apply
{rec_text}

## Your Task

Update the mutation weights based on these recommendations.
Return the new weights as valid JSON:

{{
    "mutation_weights": {{"<type>": weight, ...}},
    "crossover_rate": X,
    "rationale": "<brief explanation of changes>"
}}

Weights should sum to 1.0. Be conservative - don't make dramatic changes
unless the data strongly supports it.
"""


def compute_mutation_effectiveness(
    mutation_history: list[dict],
) -> dict[str, dict]:
    """Compute effectiveness metrics for each mutation type.

    Args:
        mutation_history: List of mutation records

    Returns:
        Dict mapping mutation type to effectiveness metrics
    """
    by_type: dict[str, list[dict]] = {}

    for m in mutation_history:
        mut_type = m.get("mutation_type", "unknown")
        if mut_type not in by_type:
            by_type[mut_type] = []
        by_type[mut_type].append(m)

    results = {}
    for mut_type, mutations in by_type.items():
        attempts = len(mutations)
        successes = sum(1 for m in mutations if m.get("success", False))

        # Calculate average impact for successful mutations
        successful = [m for m in mutations if m.get("success", False)]
        if successful:
            avg_impact = sum(m.get("fitness_delta", 0) for m in successful) / len(successful)
        else:
            avg_impact = 0.0

        success_rate = successes / attempts if attempts > 0 else 0.0
        effectiveness = success_rate * max(0, avg_impact)  # Only positive impact counts

        results[mut_type] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": round(success_rate, 3),
            "avg_impact": round(avg_impact, 3),
            "effectiveness": round(effectiveness, 3),
        }

    return results


def compute_crossover_contribution(
    crossover_history: list[dict],
) -> dict:
    """Compute crossover contribution metrics.

    Args:
        crossover_history: List of crossover records with parent and child fitness

    Returns:
        Dict with crossover contribution metrics
    """
    if not crossover_history:
        return {
            "attempts": 0,
            "wins": 0,
            "contribution_rate": 0.0,
            "avg_improvement": 0.0,
        }

    attempts = len(crossover_history)
    wins = 0
    improvements = []

    for c in crossover_history:
        child_fitness = c.get("child_fitness", 0)
        parent_fitnesses = c.get("parent_fitnesses", [])

        if parent_fitnesses:
            best_parent = max(parent_fitnesses)
            if child_fitness > best_parent:
                wins += 1
                improvements.append(child_fitness - best_parent)

    return {
        "attempts": attempts,
        "wins": wins,
        "contribution_rate": round(wins / attempts, 3) if attempts > 0 else 0.0,
        "avg_improvement": round(sum(improvements) / len(improvements), 3) if improvements else 0.0,
    }


def compute_diversity_index(
    population: list[dict],
    fitness_key: str = "fitness",
) -> dict:
    """Compute population diversity metrics.

    Args:
        population: List of population members with fitness scores
        fitness_key: Key for fitness value in population dicts

    Returns:
        Dict with diversity metrics
    """
    if not population or len(population) < 2:
        return {
            "phenotypic": 0.0,
            "population_size": len(population) if population else 0,
        }

    fitnesses = [p.get(fitness_key, 0) for p in population]

    # Phenotypic diversity: normalized standard deviation of fitness
    mean_fitness = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
    std_dev = variance ** 0.5

    # Normalize by mean (coefficient of variation)
    # Cap at 1.0 for consistency
    if mean_fitness > 0:
        phenotypic = min(1.0, std_dev / mean_fitness)
    else:
        phenotypic = 0.0

    return {
        "phenotypic": round(phenotypic, 3),
        "fitness_range": round(max(fitnesses) - min(fitnesses), 3),
        "fitness_mean": round(mean_fitness, 3),
        "fitness_std": round(std_dev, 3),
        "population_size": len(population),
    }


def should_trigger_analysis(
    generation: int,
    last_analysis_generation: int | None,
    trigger_interval: int = 5,
) -> bool:
    """Determine if meta-strategist analysis should be triggered.

    Args:
        generation: Current generation
        last_analysis_generation: Generation of last analysis (None if never)
        trigger_interval: How often to trigger (default: every 5 generations)

    Returns:
        True if analysis should be triggered
    """
    if last_analysis_generation is None:
        # First analysis after at least one full interval
        return generation >= trigger_interval

    return generation - last_analysis_generation >= trigger_interval


def _summarize_mutations(mutations: list[dict]) -> str:
    """Create a text summary of mutation history."""
    if not mutations:
        return "No mutations in analysis window."

    effectiveness = compute_mutation_effectiveness(mutations)

    lines = ["| Type | Attempts | Successes | Success Rate | Avg Impact | Effectiveness |",
             "|------|----------|-----------|--------------|------------|---------------|"]

    for mut_type, metrics in sorted(effectiveness.items(), key=lambda x: -x[1]["effectiveness"]):
        lines.append(
            f"| {mut_type} | {metrics['attempts']} | {metrics['successes']} | "
            f"{metrics['success_rate']:.1%} | {metrics['avg_impact']:+.2f} | {metrics['effectiveness']:.3f} |"
        )

    total = len(mutations)
    successful = sum(1 for m in mutations if m.get("success", False))

    lines.append("")
    lines.append(f"**Total**: {total} mutations, {successful} successful ({successful/total:.1%})")

    return "\n".join(lines)


def _summarize_crossovers(crossovers: list[dict]) -> str:
    """Create a text summary of crossover history."""
    if not crossovers:
        return "No crossovers in analysis window."

    metrics = compute_crossover_contribution(crossovers)

    return f"""- **Attempts**: {metrics['attempts']}
- **Wins** (child > best parent): {metrics['wins']}
- **Contribution Rate**: {metrics['contribution_rate']:.1%}
- **Average Improvement** (when winning): {metrics['avg_improvement']:+.2f}"""


def _summarize_fitness(fitness_history: list[dict]) -> str:
    """Create a text summary of fitness trajectory."""
    if not fitness_history:
        return "No fitness history available."

    lines = ["| Generation | Best Fitness | Avg Fitness | Improvement |",
             "|------------|--------------|-------------|-------------|"]

    prev_best = None
    for record in fitness_history:
        gen = record.get("generation", "?")
        best = record.get("best_fitness", 0)
        avg = record.get("avg_fitness", 0)

        if prev_best is not None and prev_best != 0:
            improvement = (best - prev_best) / abs(prev_best) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "-"

        lines.append(f"| {gen} | {best:.2f} | {avg:.2f} | {imp_str} |")
        prev_best = best

    # Calculate overall trend
    if len(fitness_history) >= 2:
        first_best = fitness_history[0].get("best_fitness", 0)
        last_best = fitness_history[-1].get("best_fitness", 0)
        if first_best != 0:
            overall_change = (last_best - first_best) / abs(first_best) * 100
            lines.append("")
            lines.append(f"**Overall Change**: {overall_change:+.1f}% over {len(fitness_history)} generations")

    return "\n".join(lines)


def _summarize_population(population: list[dict]) -> str:
    """Create a text summary of current population."""
    if not population:
        return "Population is empty."

    diversity = compute_diversity_index(population)

    # Sort by fitness descending
    sorted_pop = sorted(population, key=lambda x: x.get("fitness", 0), reverse=True)

    lines = [f"**Population Size**: {len(population)}",
             f"**Diversity (phenotypic)**: {diversity['phenotypic']:.2f}",
             f"**Fitness Range**: {diversity['fitness_range']:.2f}",
             "",
             "Top 5 solutions:",
             "| Rank | Solution | Fitness |",
             "|------|----------|---------|"]

    for i, p in enumerate(sorted_pop[:5], 1):
        name = p.get("file", p.get("name", "unknown"))
        fitness = p.get("fitness", 0)
        lines.append(f"| {i} | {name} | {fitness:.2f} |")

    return "\n".join(lines)
