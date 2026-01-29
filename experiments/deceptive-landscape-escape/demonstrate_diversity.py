#!/usr/bin/env python3
"""Demonstrate the Diversity Guardian agent.

This script shows how the Diversity Guardian:
1. Computes genotypic diversity (code similarity)
2. Computes phenotypic diversity (fitness spread)
3. Detects when population is converging
4. Recommends interventions

Run this to see the Diversity Guardian in action without a full evolution run.
"""

import sys
from pathlib import Path

# Add SDK to path
sdk_path = Path(__file__).parent.parent.parent / "sdk"
sys.path.insert(0, str(sdk_path))

from evolve_sdk.agents.diversity_guardian import (
    compute_genotypic_diversity,
    compute_phenotypic_diversity,
    compute_combined_diversity,
    should_alert_diversity,
)


def create_sample_population():
    """Create sample solutions demonstrating different diversity scenarios."""

    # Scenario 1: Diverse population (healthy)
    diverse_pop = [
        {"file": "gen0_a.py", "fitness": 50.0, "code": "def solve(n): return '0' * n"},
        {"file": "gen1a.py", "fitness": 65.0, "code": "def solve(n): return '1' * n"},
        {"file": "gen1b.py", "fitness": 78.0, "code": "def solve(n): return ''.join('01'[i%2] for i in range(n))"},
        {"file": "gen2a.py", "fitness": 72.0, "code": "def solve(n): return '11' * (n//2)"},
        {"file": "gen2x.py", "fitness": 85.0, "code": "def solve(n): return '10' * (n//2)"},
    ]

    # Scenario 2: Converged population (problematic)
    converged_pop = [
        {"file": "gen3a.py", "fitness": 75.2, "code": "def solve(n): return '1' * (n-2) + '00'"},
        {"file": "gen3b.py", "fitness": 75.5, "code": "def solve(n): return '1' * (n-1) + '0'"},
        {"file": "gen3c.py", "fitness": 75.3, "code": "def solve(n): return '1' * (n-3) + '000'"},
        {"file": "gen4a.py", "fitness": 75.8, "code": "def solve(n): return '0' + '1' * (n-1)"},
        {"file": "gen4x.py", "fitness": 75.6, "code": "def solve(n): return '1' * n"},
    ]

    return diverse_pop, converged_pop


def write_temp_files(population):
    """Write population code to temporary files for embedding."""
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for i, member in enumerate(population):
        file_path = os.path.join(temp_dir, f"solution_{i}.py")
        with open(file_path, "w") as f:
            f.write(member["code"])
        file_paths.append(file_path)
        member["file"] = file_path

    return temp_dir, file_paths


def demo_diversity_computation():
    """Demonstrate diversity computation."""
    print("=" * 60)
    print("DIVERSITY GUARDIAN DEMONSTRATION")
    print("=" * 60)

    diverse_pop, converged_pop = create_sample_population()

    # Demo 1: Phenotypic diversity (no embeddings needed)
    print("\n## Phenotypic Diversity (Fitness Spread)")
    print("-" * 40)

    pheno_diverse = compute_phenotypic_diversity(diverse_pop)
    pheno_converged = compute_phenotypic_diversity(converged_pop)

    print(f"Diverse population:   {pheno_diverse:.3f}")
    print(f"Converged population: {pheno_converged:.3f}")
    print(f"Threshold: 0.20")
    print()

    if pheno_converged < 0.20:
        print("-> Converged population has LOW phenotypic diversity (ALERT)")
    if pheno_diverse >= 0.20:
        print("-> Diverse population has HEALTHY phenotypic diversity (OK)")

    # Demo 2: Alert detection
    print("\n## Alert Detection")
    print("-" * 40)

    # Simulate genotypic diversity (without embeddings for demo)
    genotypic_diverse = 0.45
    genotypic_converged = 0.12

    alert_diverse, level_diverse = should_alert_diversity(genotypic_diverse, pheno_diverse)
    alert_converged, level_converged = should_alert_diversity(genotypic_converged, pheno_converged)

    print(f"Diverse population:")
    print(f"  Genotypic: {genotypic_diverse:.3f}, Phenotypic: {pheno_diverse:.3f}")
    print(f"  Alert: {alert_diverse}, Level: {level_diverse}")
    print()
    print(f"Converged population:")
    print(f"  Genotypic: {genotypic_converged:.3f}, Phenotypic: {pheno_converged:.3f}")
    print(f"  Alert: {alert_converged}, Level: {level_converged}")

    # Demo 3: Show what Diversity Guardian would do
    print("\n## Diversity Guardian Response")
    print("-" * 40)

    if alert_converged:
        print("ALERT TRIGGERED for converged population!")
        print()
        print("Diagnosis:")
        print("  - Population has converged to similar solutions")
        print("  - All solutions are 'mostly 1s' variants (local optimum trap)")
        print("  - Fitness variance is very low (everyone stuck at ~75)")
        print()
        print("Recommended Interventions:")
        print("  1. [HIGH] inject_orthogonal: Add solutions using different approaches")
        print("     - Try alternating patterns instead of high-density 1s")
        print("     - Try structured patterns like '10' repeats")
        print("  2. [MEDIUM] reduce_selection_pressure: Keep more diverse solutions")
        print("  3. [LOW] increase_mutation_variance: Encourage bolder mutations")

    print()
    print("=" * 60)
    print("END OF DEMONSTRATION")
    print("=" * 60)


def demo_with_embeddings():
    """Demonstrate with actual embeddings (requires sentence-transformers)."""
    print("\n## Genotypic Diversity (Code Embeddings)")
    print("-" * 40)

    try:
        from evolve_sdk.memory.embeddings import get_embedder
        embedder = get_embedder()

        diverse_pop, converged_pop = create_sample_population()

        # Write to temp files
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()

        for i, pop in enumerate([diverse_pop, converged_pop]):
            pop_name = "diverse" if i == 0 else "converged"
            file_paths = []

            for j, member in enumerate(pop):
                file_path = os.path.join(temp_dir, f"{pop_name}_{j}.py")
                with open(file_path, "w") as f:
                    f.write(member["code"])
                file_paths.append(file_path)
                member["file"] = file_path

            geno_div = compute_genotypic_diversity(file_paths, embedder)
            print(f"{pop_name.capitalize()} population genotypic diversity: {geno_div:.3f}")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    except ImportError:
        print("(sentence-transformers not installed - using simulated values)")
        print("Diverse population genotypic diversity: ~0.45 (simulated)")
        print("Converged population genotypic diversity: ~0.12 (simulated)")


if __name__ == "__main__":
    demo_diversity_computation()

    print("\n")
    demo_with_embeddings()
