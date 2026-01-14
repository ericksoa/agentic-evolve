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


# =============================================================================
# DIRECTION ADVISOR MODE
# Extension to Meta-Strategist for strategic direction recommendations
# =============================================================================

DIRECTION_ADVISOR_SYSTEM = """You are a strategic direction advisor for evolutionary algorithm discovery.

Your role: Analyze candidate strategic directions and recommend the best path forward
based on expected impact, risk, and historical evidence.

## Core Responsibilities

1. **Direction Analysis**: Evaluate each candidate direction's potential
2. **Impact Assessment**: Estimate expected fitness improvement for each direction
3. **Risk Assessment**: Evaluate regression risk and resource cost
4. **Historical Correlation**: Check if similar directions succeeded/failed before
5. **Ranked Recommendations**: Provide prioritized list with clear rationale

## Assessment Criteria

### Expected Impact
- **high**: Could yield >5% improvement, paradigm-shifting potential
- **medium**: Expected 1-5% improvement, solid incremental gain
- **low**: <1% expected improvement, marginal benefit

### Risk Level
- **high**: Significant regression risk, may require many generations to validate
- **medium**: Moderate risk, some exploration cost but recoverable
- **low**: Safe bet, low regression probability, quick to validate

### Confidence Score (0.0-1.0)
Based on:
- Historical evidence (similar approaches tried before)
- Theoretical soundness (does this make sense for the problem?)
- Current trajectory alignment (does this fit where evolution is heading?)

## Output Format

Always return valid JSON:
{
    "generation_analyzed": <current generation>,
    "analysis_context": {
        "current_fitness": <float>,
        "fitness_trajectory": "improving|stalled|declining",
        "plateau_status": <bool>,
        "diversity_status": "healthy|warning|critical",
        "dominant_approach": "<description of current best approach>"
    },
    "directions": [
        {
            "id": "d1",
            "name": "<short name>",
            "description": "<full description>",
            "approach_type": "algorithmic|architectural|data_structural|hybrid|domain_specific",
            "source": "generated|user_provided|breakthrough_pattern",
            "expected_impact": {
                "potential_gain": "high|medium|low",
                "confidence": <0.0-1.0>,
                "timeframe_generations": <int>
            },
            "risk_assessment": {
                "level": "high|medium|low",
                "regression_risk": <0.0-1.0>,
                "resource_cost": "high|medium|low",
                "factors": ["<risk factor 1>", "<risk factor 2>"]
            },
            "mode_fit": {
                "alignment": "strong|moderate|weak",
                "rationale": "<why this fits or doesn't fit the optimization mode>"
            },
            "evidence": {
                "similar_successes": <count from history>,
                "similar_failures": <count from history>,
                "breakthrough_correlation": <0.0-1.0>
            },
            "recommendation_score": <0-100>,
            "rationale": "<detailed reasoning for this ranking>"
        }
    ],
    "ranked_direction_ids": ["<sorted by recommendation_score descending>"],
    "recommended_action": {
        "primary_direction": "<best direction id>",
        "exploration_plan": "<how to explore this direction>",
        "fallback_direction": "<second choice if primary fails>",
        "avoid_directions": ["<directions to skip and why>"]
    },
    "guidance_for_mutators": "<specific guidance to pass to next generation mutators>"
}

Be data-driven. Justify recommendations with evidence from history and metrics.
Consider risk tolerance when ranking - conservative runs should prefer safer options.
"""


# Mode-specific direction templates
_DIRECTION_TEMPLATES = {
    "size": [
        {
            "name": "Functional Composition",
            "description": "Refactor using higher-order functions, map/filter/reduce patterns",
            "approach_type": "algorithmic",
        },
        {
            "name": "List Comprehension Exploitation",
            "description": "Replace loops with comprehensions and generator expressions",
            "approach_type": "algorithmic",
        },
        {
            "name": "Operator Overloading",
            "description": "Use operator magic methods to reduce explicit function calls",
            "approach_type": "architectural",
        },
        {
            "name": "Lambda Consolidation",
            "description": "Inline functions as lambdas where possible",
            "approach_type": "algorithmic",
        },
        {
            "name": "Import Aliasing",
            "description": "Use short aliases and from-imports to reduce character count",
            "approach_type": "data_structural",
        },
    ],
    "perf": [
        {
            "name": "Data Structure Optimization",
            "description": "Switch to more efficient data structures (sets, deques, arrays)",
            "approach_type": "data_structural",
        },
        {
            "name": "Algorithm Complexity Reduction",
            "description": "Find O(n) or O(n log n) replacements for O(n²) operations",
            "approach_type": "algorithmic",
        },
        {
            "name": "Vectorization",
            "description": "Replace loops with numpy/array operations for SIMD benefits",
            "approach_type": "architectural",
        },
        {
            "name": "Caching and Memoization",
            "description": "Add functools.lru_cache or manual memoization for repeated computations",
            "approach_type": "algorithmic",
        },
        {
            "name": "Early Termination",
            "description": "Add early exit conditions to avoid unnecessary computation",
            "approach_type": "algorithmic",
        },
    ],
    "ml": [
        {
            "name": "Feature Engineering",
            "description": "Focus on feature selection, transformation, and interaction terms",
            "approach_type": "domain_specific",
        },
        {
            "name": "Architecture Modification",
            "description": "Change model architecture (layers, connections, attention)",
            "approach_type": "architectural",
        },
        {
            "name": "Ensemble Strategies",
            "description": "Combine multiple models via voting, stacking, or blending",
            "approach_type": "hybrid",
        },
        {
            "name": "Regularization Tuning",
            "description": "Adjust dropout, weight decay, batch normalization parameters",
            "approach_type": "algorithmic",
        },
        {
            "name": "Loss Function Engineering",
            "description": "Try alternative loss functions or custom weighted objectives",
            "approach_type": "domain_specific",
        },
    ],
}


def generate_candidate_directions(
    mode: str,
    fitness_trajectory: str = "stable",
    dominant_approach: str | None = None,
    breakthrough_patterns: list[dict] | None = None,
    max_directions: int = 5,
) -> list[dict]:
    """Generate candidate strategic directions based on mode and context.

    Args:
        mode: Evolution mode (size, perf, ml)
        fitness_trajectory: Current fitness trend (improving, stalled, declining)
        dominant_approach: Description of current dominant approach
        breakthrough_patterns: Historical breakthrough patterns from memory
        max_directions: Maximum number of directions to generate

    Returns:
        List of candidate direction dictionaries
    """
    directions = []

    # Reserve slots for breakthrough patterns (up to 2)
    breakthrough_slots = min(2, len(breakthrough_patterns)) if breakthrough_patterns else 0
    template_slots = max_directions - breakthrough_slots

    # Get mode-specific templates
    templates = _DIRECTION_TEMPLATES.get(mode, _DIRECTION_TEMPLATES["perf"])

    # Select templates based on trajectory
    if fitness_trajectory == "stalled":
        # When stalled, prefer more radical directions (later in list)
        selected = templates[-template_slots:] if template_slots > 0 else []
    elif fitness_trajectory == "declining":
        # When declining, prefer safer directions (earlier in list)
        selected = templates[:template_slots]
    else:
        # When improving, mix of both
        selected = templates[:template_slots]

    for i, template in enumerate(selected):
        direction = {
            "id": f"d{i+1}",
            "name": template["name"],
            "description": template["description"],
            "approach_type": template["approach_type"],
            "source": "generated",
        }
        directions.append(direction)

    # Add breakthrough-inspired directions if available
    if breakthrough_patterns:
        for i, pattern in enumerate(breakthrough_patterns[:2]):  # Max 2 from history
            if len(directions) >= max_directions:
                break
            direction = {
                "id": f"db{i+1}",
                "name": f"Historical: {pattern.get('name', 'Breakthrough Pattern')}",
                "description": pattern.get("description", "Pattern that broke through a plateau"),
                "approach_type": pattern.get("approach_type", "hybrid"),
                "source": "breakthrough_pattern",
            }
            directions.append(direction)

    return directions[:max_directions]


def should_trigger_direction_analysis(
    generation: int,
    last_direction_analysis: int | None,
    trigger_interval: int = 10,
    tactical_stall_count: int = 0,
    tactical_stall_threshold: int = 2,
) -> bool:
    """Determine if strategic direction analysis should be triggered.

    Args:
        generation: Current generation number
        last_direction_analysis: Generation of last direction analysis (None if never)
        trigger_interval: How often to auto-trigger (default: every 10 generations)
        tactical_stall_count: Count of consecutive tactical analyses without improvement
        tactical_stall_threshold: How many stalls before triggering direction analysis

    Returns:
        True if direction analysis should be triggered
    """
    # Trigger if tactical adjustments are stalling
    if tactical_stall_count >= tactical_stall_threshold:
        return True

    # Trigger on interval
    if last_direction_analysis is None:
        # First analysis after sufficient history
        return generation >= trigger_interval
    return generation - last_direction_analysis >= trigger_interval


def get_direction_recommendation_prompt(
    generation: int,
    current_champion: dict,
    fitness_history: list[dict],
    mutation_history: list[dict],
    population_snapshot: list[dict] | None = None,
    candidate_directions: list[dict] | None = None,
    breakthrough_patterns: list[dict] | None = None,
    mode: str = "perf",
    problem_description: str = "",
    risk_tolerance: str = "medium",
    analysis_window: int = 10,
) -> str:
    """Generate prompt for strategic direction recommendation.

    Args:
        generation: Current generation number
        current_champion: Current best solution with file and fitness
        fitness_history: Historical fitness trajectory
        mutation_history: Recent mutation successes/failures
        population_snapshot: Current population for diversity assessment
        candidate_directions: User-provided directions to evaluate (optional)
        breakthrough_patterns: Historical breakthrough patterns from memory
        mode: Optimization mode (size, perf, ml)
        problem_description: Description of the problem being solved
        risk_tolerance: How aggressive to be (low, medium, high)
        analysis_window: How many generations to analyze for trends

    Returns:
        Formatted prompt for the direction advisor
    """
    mode_guidance = get_mode_guidance(mode, "meta_strategist")

    # Compute context from history
    recent_fitness = fitness_history[-(analysis_window + 1):] if len(fitness_history) > analysis_window else fitness_history

    # Determine fitness trajectory
    if len(recent_fitness) >= 2:
        first_best = recent_fitness[0].get("best_fitness", 0)
        last_best = recent_fitness[-1].get("best_fitness", 0)
        if first_best > 0:
            change_pct = (last_best - first_best) / abs(first_best) * 100
            if change_pct > 2:
                trajectory = "improving"
            elif change_pct < -2:
                trajectory = "declining"
            else:
                trajectory = "stalled"
        else:
            trajectory = "stable"
    else:
        trajectory = "stable"

    # Get champion info
    champion_fitness = current_champion.get("fitness", 0)
    champion_file = current_champion.get("file", "unknown")
    champion_approach = current_champion.get("algorithm_description", "Unknown approach")

    # Compute diversity if population available
    diversity_status = "unknown"
    if population_snapshot:
        diversity = compute_diversity_index(population_snapshot)
        phenotypic = diversity.get("phenotypic", 0.5)
        if phenotypic < 0.15:
            diversity_status = "critical"
        elif phenotypic < 0.25:
            diversity_status = "warning"
        else:
            diversity_status = "healthy"

    # Generate directions if not provided
    if not candidate_directions:
        candidate_directions = generate_candidate_directions(
            mode=mode,
            fitness_trajectory=trajectory,
            dominant_approach=champion_approach,
            breakthrough_patterns=breakthrough_patterns,
            max_directions=5,
        )

    # Format directions for prompt
    directions_text = _format_directions(candidate_directions)

    # Summarize mutation history
    recent_mutations = [m for m in mutation_history if m.get("generation", 0) >= generation - analysis_window]
    mutation_summary = _summarize_mutations(recent_mutations)

    # Summarize fitness trajectory
    fitness_summary = _summarize_fitness(recent_fitness)

    # Risk tolerance guidance
    risk_guidance = {
        "low": "Prefer safer directions with lower regression risk. Prioritize confidence over potential gain.",
        "medium": "Balance risk and reward. Accept moderate risk for meaningful potential improvement.",
        "high": "Willing to take bold bets. Prioritize high potential gain even with higher risk.",
    }.get(risk_tolerance, "Balance risk and reward.")

    prompt = f"""# Strategic Direction Analysis Request

## Problem Context
{problem_description if problem_description else "Optimization problem"}

## Current State
- **Generation**: {generation}
- **Mode**: {mode}
- **Current Champion**: {champion_file}
- **Champion Fitness**: {champion_fitness:.4f}
- **Dominant Approach**: {champion_approach}

## Evolution Trajectory
- **Fitness Trajectory**: {trajectory}
- **Diversity Status**: {diversity_status}

{mode_guidance}

## Risk Tolerance
{risk_guidance}

## Fitness History (last {analysis_window} generations)

{fitness_summary}

## Recent Mutation Patterns

{mutation_summary}

## Candidate Directions to Evaluate

{directions_text}

## Your Task

Analyze each candidate direction and provide:

1. **Impact Assessment**: Expected improvement potential and confidence
2. **Risk Assessment**: Regression risk, resource cost, risk factors
3. **Historical Evidence**: Similar successes/failures from mutation history
4. **Mode Fit**: How well this direction aligns with {mode} optimization
5. **Recommendation Score**: 0-100 composite score
6. **Ranking**: Ordered list from best to worst

Consider:
- Current fitness trajectory ({trajectory}) - what does this suggest?
- Diversity status ({diversity_status}) - do we need exploration or exploitation?
- Risk tolerance ({risk_tolerance}) - how bold should we be?
- Recent mutation patterns - what's been working or failing?

Return your analysis as valid JSON matching the output format.
"""

    return prompt


def _format_directions(directions: list[dict]) -> str:
    """Format candidate directions for the prompt."""
    if not directions:
        return "No candidate directions provided."

    lines = ["| ID | Name | Type | Description |",
             "|----|------|------|-------------|"]

    for d in directions:
        lines.append(
            f"| {d.get('id', '?')} | {d.get('name', 'Unknown')} | "
            f"{d.get('approach_type', '?')} | {d.get('description', '')} |"
        )

    return "\n".join(lines)
