"""
Pre-built query patterns for evolution memory.

Provides high-level query functions for common use cases:
- Mutation intelligence
- Breakthrough detection
- Trust calibration
- Meta-evolution
"""

from typing import Any

from .store import EvolutionMemory
from .schemas import FrameType


def get_mutation_context(
    memory: EvolutionMemory,
    parent_code: str,
    max_similar: int = 3,
    max_failed: int = 5,
) -> dict[str, Any]:
    """
    Get mutation context for a parent solution.

    Returns information to help the mutator agent:
    - Similar successful mutations
    - Failed mutations to avoid
    - High-impact patterns

    Args:
        memory: Evolution memory instance
        parent_code: Code of the parent solution
        max_similar: Maximum similar mutations to return
        max_failed: Maximum failed mutations to return

    Returns:
        Dict with mutation context for prompt injection
    """
    context = {
        "similar_successful": [],
        "failed_to_avoid": [],
        "high_impact_patterns": [],
        "suggestions": [],
    }

    # Find similar successful mutations
    similar = memory.find_similar_mutations(parent_code, limit=max_similar)
    for m in similar:
        context["similar_successful"].append({
            "diff_preview": m.get("diff_content", "")[:500],
            "fitness_delta_pct": m.get("fitness_delta_pct", 0),
            "tags": m.get("tags", []),
            "broke_plateau": m.get("broke_plateau", False),
        })

    # Find failed mutations on this parent
    failed = memory.find_failed_mutations_for_parent(parent_code, limit=max_failed)
    for f in failed:
        context["failed_to_avoid"].append({
            "diff_preview": f.get("diff_content", "")[:300],
            "reason": f.get("failure_reason", "unknown"),
        })

    # Find high-impact mutations (broke plateau or became champion)
    all_mutations = memory.query(frame_type=FrameType.MUTATION)
    high_impact = [
        m for m in all_mutations
        if m.get("broke_plateau") or m.get("became_champion")
    ]
    for m in high_impact[:3]:
        context["high_impact_patterns"].append({
            "tags": m.get("tags", []),
            "fitness_delta_pct": m.get("fitness_delta_pct", 0),
            "description": f"Gen {m.get('generation')}: +{m.get('fitness_delta_pct', 0):.1f}%",
        })

    # Generate suggestions based on patterns
    if context["similar_successful"]:
        common_tags = []
        for m in context["similar_successful"]:
            common_tags.extend(m.get("tags", []))
        if common_tags:
            context["suggestions"].append(
                f"Consider approaches tagged: {', '.join(set(common_tags)[:5])}"
            )

    if context["failed_to_avoid"]:
        context["suggestions"].append(
            f"Avoid {len(context['failed_to_avoid'])} previously failed approaches"
        )

    return context


def get_breakthrough_patterns(
    memory: EvolutionMemory,
    current_problem_description: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Find patterns that broke through plateaus.

    Useful when evolution is stuck.

    Args:
        memory: Evolution memory instance
        current_problem_description: Description of current problem for similarity
        limit: Maximum patterns to return

    Returns:
        List of breakthrough pattern descriptions
    """
    all_mutations = memory.query(frame_type=FrameType.MUTATION)
    breakthroughs = [m for m in all_mutations if m.get("broke_plateau")]

    patterns = []
    for m in breakthroughs[:limit]:
        patterns.append({
            "generation": m.get("generation"),
            "fitness_improvement": m.get("fitness_delta_pct", 0),
            "mutation_type": m.get("mutation_type", "unknown"),
            "tags": m.get("tags", []),
            "diff_preview": m.get("diff_content", "")[:300],
            "problem_id": m.get("problem_id", ""),
        })

    return patterns


def get_trust_calibration(
    memory: EvolutionMemory,
) -> dict[str, Any]:
    """
    Get trust system calibration data.

    Analyzes historical trust decisions to help calibrate thresholds.

    Returns:
        Dict with calibration statistics and recommendations
    """
    trust_decisions = memory.query(frame_type=FrameType.TRUST_DECISION)
    human_decisions = memory.query(frame_type=FrameType.HUMAN_DECISION)

    calibration = {
        "total_trust_decisions": len(trust_decisions),
        "total_human_overrides": len(human_decisions),
        "recommendation_breakdown": {},
        "human_override_rate": 0,
        "false_positives": [],  # Rejected but human accepted
        "suggestions": [],
    }

    # Breakdown by recommendation
    for d in trust_decisions:
        rec = d.get("recommendation", "unknown")
        calibration["recommendation_breakdown"][rec] = (
            calibration["recommendation_breakdown"].get(rec, 0) + 1
        )

    # Calculate human override rate
    if trust_decisions:
        rejected = [d for d in trust_decisions if d.get("recommendation") == "reject"]
        overridden = [
            h for h in human_decisions
            if h.get("human_decision") == "accept"
            and h.get("adversary_recommendation") == "reject"
        ]
        if rejected:
            calibration["human_override_rate"] = len(overridden) / len(rejected)

    # Identify false positives (rejected by system, accepted by human)
    for h in human_decisions:
        if h.get("human_decision") == "accept" and h.get("adversary_recommendation") == "reject":
            calibration["false_positives"].append({
                "file": h.get("file_path"),
                "original_trust": h.get("original_trust"),
                "flags": h.get("flags", []),
            })

    # Generate suggestions
    if calibration["human_override_rate"] > 0.3:
        calibration["suggestions"].append(
            f"High override rate ({calibration['human_override_rate']:.0%}). "
            "Consider lowering reject_threshold."
        )

    if len(calibration["false_positives"]) > 5:
        calibration["suggestions"].append(
            f"{len(calibration['false_positives'])} false positives detected. "
            "Review flag patterns that trigger unnecessary rejections."
        )

    return calibration


def get_exploit_patterns(
    memory: EvolutionMemory,
) -> dict[str, Any]:
    """
    Get summary of known exploit patterns.

    Useful for adversary agent context.

    Returns:
        Dict with exploit pattern summary
    """
    exploits = memory.query(frame_type=FrameType.EXPLOIT)

    summary = {
        "total_exploits": len(exploits),
        "pattern_types": {},
        "common_flags": {},
        "examples": [],
    }

    for e in exploits:
        # Count pattern types
        pattern = e.get("pattern_type", "unknown")
        summary["pattern_types"][pattern] = summary["pattern_types"].get(pattern, 0) + 1

        # Count common flags
        for flag in e.get("flags", []):
            summary["common_flags"][flag] = summary["common_flags"].get(flag, 0) + 1

    # Get examples (most recent)
    for e in exploits[-3:]:
        summary["examples"].append({
            "pattern_type": e.get("pattern_type"),
            "flags": e.get("flags", []),
            "code_preview": e.get("code_content", "")[:200],
        })

    return summary


def get_meta_evolution_insights(
    memory: EvolutionMemory,
    mode: str = "",
) -> dict[str, Any]:
    """
    Get insights about successful evolution configurations.

    For meta-evolution: learning which configs work best.

    Args:
        memory: Evolution memory instance
        mode: Filter by mode (size, perf, ml)

    Returns:
        Dict with config insights
    """
    configs = memory.query(frame_type=FrameType.CONFIG)
    if mode:
        configs = [c for c in configs if c.get("mode") == mode]

    insights = {
        "total_runs": len(configs),
        "successful_runs": 0,
        "avg_improvement": 0,
        "best_settings": {},
        "recommendations": [],
    }

    if not configs:
        return insights

    successful = [c for c in configs if c.get("success")]
    insights["successful_runs"] = len(successful)

    if successful:
        avg_improvement = sum(c.get("improvement_pct", 0) for c in successful) / len(successful)
        insights["avg_improvement"] = avg_improvement

        # Find common settings in successful runs
        settings_frequency = {}
        for c in successful:
            for key, value in c.get("key_settings", {}).items():
                setting = f"{key}={value}"
                settings_frequency[setting] = settings_frequency.get(setting, 0) + 1

        # Top settings
        sorted_settings = sorted(
            settings_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        insights["best_settings"] = dict(sorted_settings[:5])

        # Recommendations
        if sorted_settings:
            top_setting, count = sorted_settings[0]
            if count > len(successful) * 0.5:
                insights["recommendations"].append(
                    f"Consider using {top_setting} (used in {count}/{len(successful)} successful runs)"
                )

    return insights


def format_mutation_context_for_prompt(context: dict[str, Any]) -> str:
    """
    Format mutation context as text for prompt injection.

    Args:
        context: Context from get_mutation_context()

    Returns:
        Formatted text for including in mutator prompt
    """
    lines = []

    if context.get("similar_successful"):
        lines.append("## Successful patterns from similar code:")
        for i, m in enumerate(context["similar_successful"][:3], 1):
            delta = m.get("fitness_delta_pct", 0)
            tags = ", ".join(m.get("tags", [])) or "none"
            lines.append(f"  {i}. +{delta:.1f}% improvement (tags: {tags})")
        lines.append("")

    if context.get("failed_to_avoid"):
        lines.append("## Approaches to AVOID (already tried and failed):")
        for i, f in enumerate(context["failed_to_avoid"][:3], 1):
            reason = f.get("reason", "unknown")
            lines.append(f"  {i}. Failed due to: {reason}")
        lines.append("")

    if context.get("high_impact_patterns"):
        lines.append("## High-impact patterns that broke plateaus:")
        for p in context["high_impact_patterns"][:2]:
            tags = ", ".join(p.get("tags", [])) or "none"
            lines.append(f"  - {p.get('description')} (tags: {tags})")
        lines.append("")

    if context.get("suggestions"):
        lines.append("## Suggestions:")
        for s in context["suggestions"]:
            lines.append(f"  - {s}")

    return "\n".join(lines) if lines else ""
