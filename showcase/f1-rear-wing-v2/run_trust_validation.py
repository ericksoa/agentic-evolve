#!/usr/bin/env python3
"""Run trust system validation on F1 wing mutations."""

import sys
import json
import time
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk"))

from evolve_sdk.trust import (
    TrustDossier,
    VarianceGate,
    ExploitDetector,
    CanaryTest,
)


def evaluate_wing(solution_path: str) -> tuple[float, float]:
    """Evaluate a wing solution and return (fitness, time_ms)."""
    import subprocess

    start = time.perf_counter()
    result = subprocess.run(
        [
            "/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/f1-rear-wing-evolution/.venv/bin/python3",
            "evaluate_physics.py",
            solution_path,
            "--json",
            "--circuit=silverstone"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    try:
        data = json.loads(result.stdout)
        return data.get("fitness", 0), elapsed_ms
    except:
        return 0.0, elapsed_ms


def main():
    work_dir = Path(__file__).parent
    mutations_dir = work_dir / "mutations"

    print("=" * 60)
    print("F1 Wing Evolution v2 - Trust System Validation")
    print("=" * 60)
    print()

    # Initialize trust components
    dossier = TrustDossier(work_dir, "F1 rear wing evolution v2")

    variance_gate = VarianceGate(
        n_evaluations=3,
        variance_threshold=0.05,
        require_gate=False,
    )

    exploit_detector = ExploitDetector(
        check_timing=True,
        timing_threshold_ms=50.0,
        check_output=True,
        output_max_value=1000.0,
        check_determinism=True,
    )

    # Run canary test first
    print("[CANARY TEST]")
    print("-" * 40)
    canary = CanaryTest(strict_mode=False)

    def simple_trust_eval(eval_result):
        fitness = eval_result.get("fitness", 0)
        flags = []

        # Check for extreme values
        if fitness > 1000:
            flags.append("Extreme fitness")
        if fitness < 0:
            flags.append("Negative fitness")

        import math
        if isinstance(fitness, float) and math.isnan(fitness):
            flags.append("NaN fitness")

        trust_score = max(0.0, 1.0 - len(flags) * 0.4)
        recommendation = "reject" if trust_score < 0.4 or len(flags) >= 2 else "accept"

        return {
            "trust_score": trust_score,
            "recommendation": recommendation,
            "flags": flags,
        }

    canary_results = canary.run_all_canaries(
        output_dir=work_dir / "canary_tests",
        trust_evaluate_fn=simple_trust_eval,
        mode="perf",
    )

    for r in canary_results:
        status = "PASS" if r.system_working else "FAIL"
        print(f"  [{status}] {r.canary_type}: trust={r.canary_trust:.2f}, rejected={r.canary_rejected}")
        dossier.record_canary_result(
            canary_fitness=r.canary_fitness,
            canary_trust=r.canary_trust,
            canary_rejected=r.canary_rejected,
            canary_flags=r.canary_flags,
        )

    print()

    # Evaluate baseline
    print("[BASELINE EVALUATION]")
    print("-" * 40)
    baseline_fitness, baseline_time = evaluate_wing("baseline_physics.json")
    print(f"  Baseline fitness: {baseline_fitness:.2f}")
    print(f"  Evaluation time: {baseline_time:.1f}ms")
    print()

    # Evaluate mutations
    print("[MUTATION EVALUATION + TRUST CHECKS]")
    print("-" * 40)

    results = []

    for mutation_file in sorted(mutations_dir.glob("gen1_*.json")):
        name = mutation_file.stem
        print(f"\n  === {name} ===")

        # Run 3 evaluations for variance gate
        fitness_values = []
        times = []
        for i in range(3):
            fitness, eval_time = evaluate_wing(str(mutation_file))
            fitness_values.append(fitness)
            times.append(eval_time)

        avg_fitness = sum(fitness_values) / len(fitness_values)
        avg_time = sum(times) / len(times)

        # Variance gate
        variance_stats = variance_gate.check_consistency(fitness_values)
        variance_flags = variance_gate.get_flags(variance_stats)

        # Exploit detection
        exploit_results = exploit_detector.run_all_checks(
            fitness=avg_fitness,
            evaluation_time_ms=avg_time,
            fitness_values=fitness_values,
        )
        exploit_flags = exploit_detector.get_flags(exploit_results)

        # Calculate improvement
        improvement_pct = ((avg_fitness - baseline_fitness) / baseline_fitness * 100) if baseline_fitness > 0 else 0
        suspicious_jump = improvement_pct > 15

        # Determine trust
        all_flags = variance_flags + exploit_flags
        if suspicious_jump:
            all_flags.append(f"Suspicious jump: {improvement_pct:.1f}%")

        trust_score = max(0.0, 1.0 - len(all_flags) * 0.15)

        # Recommendation
        if avg_fitness <= 0:
            recommendation = "reject"
            trust_score = 0
        elif exploit_detector.has_critical(exploit_results):
            recommendation = "reject"
            trust_score = 0
        elif len(all_flags) >= 3:
            recommendation = "challenge"
        else:
            recommendation = "accept"

        print(f"    Fitness: {avg_fitness:.2f} (±{variance_stats.std:.2f})")
        print(f"    Improvement: {improvement_pct:+.1f}%")
        print(f"    Variance CV: {variance_stats.cv:.4f} {'PASS' if variance_stats.passed else 'FAIL'}")
        print(f"    Eval time: {avg_time:.1f}ms")
        print(f"    Exploit checks: {len(exploit_flags)} flags")
        print(f"    Trust score: {trust_score:.2f}")
        print(f"    Recommendation: {recommendation.upper()}")

        if all_flags:
            print(f"    Flags: {', '.join(all_flags)}")

        # Record in dossier
        dossier.record_evaluation(
            candidate_file=str(mutation_file),
            generation=1,
            fitness=avg_fitness,
            trust_score=trust_score,
            recommendation=recommendation,
            flags=all_flags,
            variance_stats=variance_stats.to_dict(),
            exploit_checks={k: v.to_dict() for k, v in exploit_results.items()},
            analysis=f"Improvement: {improvement_pct:+.1f}% from baseline",
        )

        results.append({
            "name": name,
            "fitness": avg_fitness,
            "trust_score": trust_score,
            "recommendation": recommendation,
            "improvement_pct": improvement_pct,
        })

    # Find champion
    print("\n" + "=" * 60)
    print("[CHAMPION SELECTION]")
    print("-" * 40)

    accepted = [r for r in results if r["recommendation"] == "accept"]
    if accepted:
        champion = max(accepted, key=lambda x: x["fitness"] * x["trust_score"])
        print(f"  Champion: {champion['name']}")
        print(f"  Fitness: {champion['fitness']:.2f}")
        print(f"  Trust-adjusted: {champion['fitness'] * champion['trust_score']:.2f}")
        print(f"  Trust score: {champion['trust_score']:.2f}")
        print(f"  Improvement: {champion['improvement_pct']:+.1f}%")

        dossier.record_champion_decision(
            old_champion={"file": "baseline_physics.json", "fitness": baseline_fitness},
            new_champion={"file": champion["name"], "fitness": champion["fitness"]},
            decision="promoted",
            reason=f"Best trust-adjusted fitness ({champion['fitness'] * champion['trust_score']:.2f})",
        )
    else:
        print("  No acceptable candidates found!")

    # Save dossier
    print("\n" + "=" * 60)
    print("[TRUST DOSSIER]")
    print("-" * 40)
    dossier.save(include_history=True)
    print(f"  Saved to: {dossier.dossier_path}")

    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("-" * 40)
    print(f"  Baseline: {baseline_fitness:.2f}")
    print(f"  Best mutation: {max(r['fitness'] for r in results):.2f}")
    print(f"  Canary tests: {sum(1 for r in canary_results if r.system_working)}/{len(canary_results)} passed")
    print(f"  Trust dossier generated: Yes")


if __name__ == "__main__":
    main()
