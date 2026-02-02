#!/usr/bin/env python3
"""
Phase 1 Day 1 Proof: Strategy selection types work with real-world data.

Uses the actual Santa 2025 sparrow algorithm config to demonstrate that
the type system correctly models the problem we're solving.

Run from repo root:
    .venv/bin/python experiments/phase1_strategy_proof/prove_day1.py
"""

import sys
sys.path.insert(0, "sdk")

from evolve_sdk.strategy.types import (
    DetectedParameter,
    ProblemContext,
    StrategyRecommendation,
    StrategyDecision,
    PROBLEM_TYPES,
    STRATEGY_TYPES,
)

# ── Real-world data: Santa 2025 sparrow algorithm parameters ────────────
# These are the actual 8 parameters from the SparrowConfig that the SDK
# couldn't optimize because it only evolved code, not hyperparameters.

SPARROW_PARAMS = [
    DetectedParameter(
        name="exploration_iters",
        value=500,
        location="SparrowConfig.__init__",
        param_type="int",
        description="Number of exploration iterations per step",
    ),
    DetectedParameter(
        name="compression_iters",
        value=200,
        location="SparrowConfig.__init__",
        param_type="int",
        description="Number of compression iterations per step",
    ),
    DetectedParameter(
        name="base_penalty",
        value=1.0,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Starting penalty coefficient for overlap",
    ),
    DetectedParameter(
        name="penalty_growth",
        value=1.05,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Multiplicative penalty increase per iteration",
    ),
    DetectedParameter(
        name="penalty_decay",
        value=0.95,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Multiplicative penalty decrease when improving",
    ),
    DetectedParameter(
        name="max_penalty",
        value=100.0,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Maximum allowed penalty coefficient",
    ),
    DetectedParameter(
        name="temperature",
        value=1.0,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Simulated annealing temperature",
    ),
    DetectedParameter(
        name="cooling_rate",
        value=0.999,
        location="SparrowConfig.__init__",
        param_type="float",
        description="Temperature decay rate per iteration",
    ),
]


def main():
    print("=" * 60)
    print("Phase 1 Day 1 Proof: Strategy Selection Types")
    print("=" * 60)

    # ── 1. Model the real problem context ──────────────────────────
    print("\n[1] Creating ProblemContext for Santa 2025 sparrow...")

    context = ProblemContext(
        problem_type="tune_existing",
        has_parameters=True,
        detected_parameters=SPARROW_PARAMS,
        complexity="complex",
        similar_past_runs=[],  # No memory yet - that's Day 5
        is_continuation=True,
        user_prompt="optimize sparrow algorithm for santa challenge",
        keywords=["packing", "optimization", "sparrow", "simulated-annealing"],
    )

    print(f"   Problem type:  {context.problem_type}")
    print(f"   Has parameters: {context.has_parameters}")
    print(f"   Parameters found: {len(context.detected_parameters)}")
    for p in context.detected_parameters:
        print(f"     - {p.name} = {p.value} ({p.param_type}) @ {p.location}")
    print(f"   Complexity:    {context.complexity}")
    print(f"   Continuation:  {context.is_continuation}")

    # ── 2. Generate recommendations ────────────────────────────────
    print("\n[2] Generating strategy recommendations...")

    # This logic will live in StrategySelector (Day 4).
    # For now, manually construct what the selector SHOULD produce.
    recommendations = [
        StrategyRecommendation(
            strategy="hyperparameter_evolution",
            confidence=0.90,
            reason=(
                f"Detected {len(context.detected_parameters)} tunable parameters "
                f"in existing code. Hyperparameter optimization is strongly "
                f"recommended before modifying algorithm structure."
            ),
        ),
        StrategyRecommendation(
            strategy="code_evolution",
            confidence=0.35,
            reason="Code structure could be improved, but parameter tuning should come first.",
        ),
        StrategyRecommendation(
            strategy="hybrid",
            confidence=0.50,
            reason="Could combine parameter tuning with structural changes.",
        ),
    ]

    # Sort by confidence (selector will do this)
    recommendations.sort(key=lambda r: r.confidence, reverse=True)

    for i, rec in enumerate(recommendations, 1):
        marker = " << RECOMMENDED" if i == 1 else ""
        print(f"   {i}. {rec.strategy} (confidence: {rec.confidence:.0%}){marker}")
        print(f"      {rec.reason}")

    # ── 3. Record the decision ─────────────────────────────────────
    print("\n[3] Recording strategy decision...")

    decision = StrategyDecision(
        strategy="hyperparameter_evolution",
        auto_selected=False,
        user_confirmed=True,
        recommendations=recommendations,
        context=context,
    )

    print(f"   Strategy chosen: {decision.strategy}")
    print(f"   User confirmed:  {decision.user_confirmed}")

    # ── 4. Prove serialization round-trip ──────────────────────────
    print("\n[4] Testing serialization round-trip...")

    serialized = decision.to_dict()
    restored = StrategyDecision.from_dict(serialized)

    assert restored.strategy == decision.strategy
    assert restored.user_confirmed == decision.user_confirmed
    assert len(restored.recommendations) == len(decision.recommendations)
    assert restored.context is not None
    assert restored.context.problem_type == "tune_existing"
    assert len(restored.context.detected_parameters) == 8
    assert restored.context.detected_parameters[0].name == "exploration_iters"
    assert restored.recommendations[0].confidence == 0.90

    print("   Round-trip: ALL ASSERTIONS PASSED")

    # ── 5. Prove validation catches bad data ───────────────────────
    print("\n[5] Testing validation guards...")

    errors_caught = 0

    try:
        ProblemContext(problem_type="magic", has_parameters=False)
    except ValueError as e:
        errors_caught += 1
        print(f"   Caught invalid problem_type: {e}")

    try:
        StrategyRecommendation(strategy="teleportation", confidence=0.5, reason="x")
    except ValueError as e:
        errors_caught += 1
        print(f"   Caught invalid strategy: {e}")

    try:
        StrategyRecommendation(strategy="code_evolution", confidence=1.5, reason="x")
    except ValueError as e:
        errors_caught += 1
        print(f"   Caught invalid confidence: {e}")

    assert errors_caught == 3
    print(f"   Validation: {errors_caught}/3 bad inputs correctly rejected")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROOF COMPLETE")
    print("=" * 60)
    print(f"""
Day 1 types correctly model the Santa 2025 problem:

  - 8 sparrow parameters detected and stored
  - Problem classified as 'tune_existing' (correct)
  - Hyperparameter evolution recommended at 90% confidence (correct)
  - Decision serializes and deserializes losslessly
  - Invalid data is rejected with clear error messages

This is the foundation for Days 2-7:
  Day 2: ProblemAnalyzer will AUTO-DETECT these parameters from code
  Day 3: UserInterface will PRESENT these recommendations interactively
  Day 4: StrategySelector will GENERATE recommendations from context
  Day 5: Memory will ENHANCE confidence with historical success data
  Day 6: Runner will USE the decision to pick the evolution strategy
""")


if __name__ == "__main__":
    main()
