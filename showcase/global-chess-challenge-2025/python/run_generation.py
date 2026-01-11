"""
Evolution Generation Runner for Global Chess Challenge 2025

Orchestrates a single generation of evolution:
1. Evaluate all candidates in current population
2. Select winners and generate offspring
3. Document results with visualizations
4. Update evolution state

Usage:
    python run_generation.py [--generation N]
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import ChessModelEvaluator, generate_test_positions, demo_random_model
from visualize import (
    CandidateResult,
    generate_fitness_bar_chart,
    generate_trajectory_chart,
    generate_results_markdown
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
EVOLVE_SDK_DIR = PROJECT_ROOT / ".evolve-sdk" / "evolve_chess_training_strategy"
MUTATIONS_DIR = EVOLVE_SDK_DIR / "mutations"
EVOLUTION_JSON = EVOLVE_SDK_DIR / "evolution.json"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_evolution_state() -> Dict[str, Any]:
    """Load current evolution state"""
    with open(EVOLUTION_JSON, 'r') as f:
        return json.load(f)


def save_evolution_state(state: Dict[str, Any]):
    """Save evolution state"""
    state["updated_at"] = datetime.now().isoformat()
    with open(EVOLUTION_JSON, 'w') as f:
        json.dump(state, f, indent=2)


def load_candidate_config(filepath: str) -> Dict[str, Any]:
    """Load a candidate configuration"""
    full_path = EVOLVE_SDK_DIR / filepath
    with open(full_path, 'r') as f:
        return json.load(f)


def save_candidate_config(config: Dict[str, Any], filepath: str):
    """Save a candidate configuration"""
    full_path = EVOLVE_SDK_DIR / filepath
    with open(full_path, 'w') as f:
        json.dump(config, f, indent=2)


def evaluate_candidate(
    config: Dict[str, Any],
    test_positions: int = 50,
    depth: int = 15
) -> Dict[str, float]:
    """
    Evaluate a candidate training configuration.

    In a full implementation, this would:
    1. Generate training data using config
    2. Fine-tune a model
    3. Evaluate the model

    For now, we simulate with a heuristic based on config quality.
    """
    # Simulated evaluation based on config heuristics
    # In production, this would actually train and evaluate a model

    base_acpl = 200  # Starting baseline

    data_config = config.get("data_config", {})
    training_config = config.get("training_config", {})

    # Heuristics for simulation (replace with actual training later)
    adjustments = 0

    # Higher min_elo = better quality data
    min_elo = data_config.get("min_elo", 1800)
    if min_elo >= 2200:
        adjustments -= 40
    elif min_elo >= 2000:
        adjustments -= 20

    # Balanced game phases
    mid_frac = data_config.get("middlegame_fraction", 0.5)
    if 0.4 <= mid_frac <= 0.6:
        adjustments -= 15

    # Deeper stockfish analysis = cleaner labels
    sf_depth = data_config.get("stockfish_depth", 18)
    if sf_depth >= 20:
        adjustments -= 10

    # Training parameters
    lr = training_config.get("learning_rate", 2e-5)
    epochs = training_config.get("num_epochs", 3)

    if 1e-5 <= lr <= 3e-5:
        adjustments -= 10
    if 3 <= epochs <= 5:
        adjustments -= 10

    # LoRA configuration
    lora_rank = training_config.get("lora_rank", 32)
    if lora_rank >= 48:
        adjustments -= 5

    # Add some noise for realism
    noise = random.gauss(0, 15)

    final_acpl = max(50, base_acpl + adjustments + noise)

    # Simulate other metrics
    legal_rate = min(1.0, 0.95 + random.gauss(0, 0.02))
    best_rate = max(0, min(1.0, (200 - final_acpl) / 500 + random.gauss(0, 0.05)))

    return {
        "acpl": final_acpl,
        "legal_rate": legal_rate,
        "best_move_rate": best_rate,
        "valid": True,
        "error": None
    }


def mutate_config(
    parent_config: Dict[str, Any],
    mutation_id: str,
    mutation_type: str = "mutation"
) -> Dict[str, Any]:
    """
    Create a mutated version of a parent config.

    Mutations include:
    - Adjusting Elo range
    - Changing game phase distribution
    - Modifying training hyperparameters
    - Altering prompt format
    """
    import copy
    config = copy.deepcopy(parent_config)

    config["id"] = mutation_id
    config["generation"] = parent_config.get("generation", 0) + 1
    config["parent"] = parent_config.get("id", "unknown")
    config["mutation_type"] = mutation_type
    config["fitness"] = None
    config["valid"] = None

    data_config = config.get("data_config", {})
    training_config = config.get("training_config", {})

    # Random mutations
    mutation_choice = random.choice([
        "elo_range", "game_phases", "learning_rate",
        "epochs", "stockfish_depth", "lora_rank"
    ])

    if mutation_choice == "elo_range":
        delta = random.choice([-200, -100, 100, 200])
        data_config["min_elo"] = max(1400, min(2400, data_config.get("min_elo", 1800) + delta))
        config["description"] = f"Mutated min_elo by {delta}"
        config["hypothesis"] = f"{'Higher' if delta > 0 else 'Lower'} Elo threshold may improve data quality"

    elif mutation_choice == "game_phases":
        # Shift focus between phases
        shift = random.choice(["more_opening", "more_middlegame", "more_endgame"])
        if shift == "more_opening":
            data_config["opening_fraction"] = min(0.5, data_config.get("opening_fraction", 0.25) + 0.1)
            data_config["endgame_fraction"] = max(0.1, data_config.get("endgame_fraction", 0.25) - 0.1)
        elif shift == "more_middlegame":
            data_config["middlegame_fraction"] = min(0.7, data_config.get("middlegame_fraction", 0.5) + 0.1)
        else:
            data_config["endgame_fraction"] = min(0.5, data_config.get("endgame_fraction", 0.25) + 0.1)
            data_config["opening_fraction"] = max(0.1, data_config.get("opening_fraction", 0.25) - 0.1)
        config["description"] = f"Shifted game phase focus: {shift}"
        config["hypothesis"] = f"More {shift.replace('more_', '')} positions may improve pattern learning"

    elif mutation_choice == "learning_rate":
        factor = random.choice([0.5, 0.7, 1.5, 2.0])
        training_config["learning_rate"] = training_config.get("learning_rate", 2e-5) * factor
        config["description"] = f"Adjusted learning rate by {factor}x"
        config["hypothesis"] = f"{'Lower' if factor < 1 else 'Higher'} LR may {'prevent overfitting' if factor < 1 else 'speed convergence'}"

    elif mutation_choice == "epochs":
        delta = random.choice([-1, 1, 2])
        training_config["num_epochs"] = max(1, min(8, training_config.get("num_epochs", 3) + delta))
        config["description"] = f"Adjusted epochs by {delta}"
        config["hypothesis"] = f"{'More' if delta > 0 else 'Fewer'} epochs may {'improve' if delta > 0 else 'prevent overfitting with'} learning"

    elif mutation_choice == "stockfish_depth":
        delta = random.choice([-4, -2, 2, 4])
        data_config["stockfish_depth"] = max(10, min(25, data_config.get("stockfish_depth", 18) + delta))
        config["description"] = f"Adjusted Stockfish depth by {delta}"
        config["hypothesis"] = f"{'Deeper' if delta > 0 else 'Shallower'} analysis may {'improve label quality' if delta > 0 else 'reduce computation time'}"

    elif mutation_choice == "lora_rank":
        delta = random.choice([-16, 16, 32])
        training_config["lora_rank"] = max(8, min(128, training_config.get("lora_rank", 32) + delta))
        training_config["lora_alpha"] = training_config["lora_rank"] * 2
        config["description"] = f"Adjusted LoRA rank by {delta}"
        config["hypothesis"] = f"{'Higher' if delta > 0 else 'Lower'} LoRA rank may {'increase' if delta > 0 else 'regularize'} model capacity"

    config["data_config"] = data_config
    config["training_config"] = training_config

    return config


def crossover_configs(
    parent1: Dict[str, Any],
    parent2: Dict[str, Any],
    crossover_id: str
) -> Dict[str, Any]:
    """
    Create a crossover of two parent configs.

    Takes best elements from each parent.
    """
    import copy

    # Start with parent1
    config = copy.deepcopy(parent1)

    config["id"] = crossover_id
    config["generation"] = max(
        parent1.get("generation", 0),
        parent2.get("generation", 0)
    ) + 1
    config["parents"] = [parent1.get("id", "unknown"), parent2.get("id", "unknown")]
    config["mutation_type"] = "crossover"
    config["fitness"] = None
    config["valid"] = None

    # Mix data configs
    data1 = parent1.get("data_config", {})
    data2 = parent2.get("data_config", {})

    config["data_config"] = {
        "min_elo": random.choice([data1.get("min_elo", 1800), data2.get("min_elo", 1800)]),
        "max_elo": max(data1.get("max_elo", 2800), data2.get("max_elo", 2800)),
        "opening_fraction": (data1.get("opening_fraction", 0.25) + data2.get("opening_fraction", 0.25)) / 2,
        "middlegame_fraction": (data1.get("middlegame_fraction", 0.5) + data2.get("middlegame_fraction", 0.5)) / 2,
        "endgame_fraction": (data1.get("endgame_fraction", 0.25) + data2.get("endgame_fraction", 0.25)) / 2,
        "stockfish_depth": max(data1.get("stockfish_depth", 18), data2.get("stockfish_depth", 18)),
        "mirror_positions": data1.get("mirror_positions", True) or data2.get("mirror_positions", True),
        "both_colors": True,
        "max_examples": max(data1.get("max_examples", 50000), data2.get("max_examples", 50000)),
    }

    # Mix training configs
    train1 = parent1.get("training_config", {})
    train2 = parent2.get("training_config", {})

    config["training_config"] = {
        "base_model": train1.get("base_model", "Qwen/Qwen2.5-3B"),
        "learning_rate": (train1.get("learning_rate", 2e-5) + train2.get("learning_rate", 2e-5)) / 2,
        "num_epochs": random.choice([train1.get("num_epochs", 3), train2.get("num_epochs", 3)]),
        "batch_size": min(train1.get("batch_size", 8), train2.get("batch_size", 8)),
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lora_rank": max(train1.get("lora_rank", 32), train2.get("lora_rank", 32)),
        "lora_alpha": max(train1.get("lora_alpha", 64), train2.get("lora_alpha", 64)),
    }

    config["description"] = f"Crossover of {parent1.get('id')} and {parent2.get('id')}"
    config["hypothesis"] = "Combining best traits from both parents may yield superior results"

    # Mix prompt formats
    if "prompt_format" in parent1 and "prompt_format" in parent2:
        config["prompt_format"] = random.choice([
            parent1["prompt_format"],
            parent2["prompt_format"]
        ])

    return config


def analyze_results(
    candidates: List[CandidateResult],
    configs: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Analyze generation results to extract learnings.

    Returns dict with:
    - worked: strategies that improved
    - failed: strategies that didn't work
    - learnings: key insights
    - next_strategy: suggestions for next generation
    """
    sorted_by_fitness = sorted(
        [(c, cfg) for c, cfg in zip(candidates, configs) if c.is_valid],
        key=lambda x: x[0].acpl
    )

    worked = []
    failed = []
    learnings = []
    next_strategy = []

    if not sorted_by_fitness:
        return {
            "worked": ["No valid candidates to analyze"],
            "failed": ["All candidates failed validation"],
            "learnings": ["Need to debug evaluation pipeline"],
            "next_strategy": ["Fix validation issues before continuing"],
        }

    best = sorted_by_fitness[0]
    worst = sorted_by_fitness[-1]

    # Analyze what made best candidate good
    best_cfg = best[1]
    worst_cfg = worst[1]

    best_data = best_cfg.get("data_config", {})
    worst_data = worst_cfg.get("data_config", {})

    # Elo analysis
    if best_data.get("min_elo", 0) > worst_data.get("min_elo", 0):
        worked.append(f"Higher Elo threshold ({best_data.get('min_elo')}) produced better results")
        learnings.append("Quality of training data (higher Elo) matters more than quantity")
    else:
        learnings.append("Lower Elo threshold can work if combined with other strong parameters")

    # Phase analysis
    best_mid = best_data.get("middlegame_fraction", 0.5)
    if 0.4 <= best_mid <= 0.6:
        worked.append(f"Balanced middlegame focus ({best_mid:.0%}) worked well")
    elif best_mid > 0.6:
        worked.append(f"Heavy middlegame focus ({best_mid:.0%}) improved tactical recognition")

    # Failed strategies
    if len(sorted_by_fitness) >= 2:
        for cand, cfg in sorted_by_fitness[-2:]:
            if cand.acpl > best[0].acpl * 1.5:
                failed.append(f"{cfg.get('id')}: {cfg.get('description', 'Unknown strategy')}")

    # Next generation strategy
    next_strategy.append(f"Mutate champion ({best_cfg.get('id')}) with variations on: {best_cfg.get('description', 'current approach')}")
    if len(sorted_by_fitness) >= 2:
        next_strategy.append(f"Crossover top 2 candidates to combine their strengths")

    return {
        "worked": worked if worked else ["No clear winning strategies identified"],
        "failed": failed if failed else ["No catastrophic failures"],
        "learnings": learnings if learnings else ["Need more generations to identify patterns"],
        "next_strategy": next_strategy,
    }


def run_generation(generation: Optional[int] = None):
    """
    Run a single generation of evolution.

    1. Load state
    2. Evaluate population
    3. Select and reproduce
    4. Document results
    5. Save state
    """
    print("=" * 60)
    print("EVOLUTION GENERATION RUNNER")
    print("=" * 60)

    # Create output directories
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load state
    state = load_evolution_state()
    current_gen = generation if generation is not None else state["generation"]

    print(f"\n📊 Running Generation {current_gen}")
    print(f"   Population size: {len(state['population'])}")

    # Evaluate all candidates
    print("\n🔬 Phase 1: Evaluating candidates...")
    candidates = []
    configs = []

    for i, pop_entry in enumerate(state["population"]):
        print(f"   [{i+1}/{len(state['population'])}] Evaluating {pop_entry['file']}...", end=" ")

        config = load_candidate_config(pop_entry["file"])
        configs.append(config)

        result = evaluate_candidate(config)

        # Update config with results
        config["fitness"] = result["acpl"]
        config["valid"] = result["valid"]
        config["error"] = result["error"]
        config["notes"] = f"ACPL={result['acpl']:.1f}, Legal={result['legal_rate']*100:.1f}%, Best={result['best_move_rate']*100:.1f}%"

        save_candidate_config(config, pop_entry["file"])

        # Update population entry
        pop_entry["fitness"] = result["acpl"]

        candidates.append(CandidateResult(
            id=config.get("id", f"candidate_{i}"),
            generation=current_gen,
            acpl=result["acpl"],
            legal_rate=result["legal_rate"],
            best_rate=result["best_move_rate"],
            is_champion=False,
            is_valid=result["valid"],
            strategy=config.get("description", "unknown")
        ))

        print(f"ACPL={result['acpl']:.1f}")

    # Determine champion
    valid_candidates = [(c, cfg) for c, cfg in zip(candidates, configs) if c.is_valid]
    if valid_candidates:
        best_candidate, best_config = min(valid_candidates, key=lambda x: x[0].acpl)
        best_candidate.is_champion = True

        # Update global champion
        prev_champion = state.get("champion") or {}
        prev_champion_acpl = prev_champion.get("fitness", float("inf")) if prev_champion else float("inf")
        if best_candidate.acpl < prev_champion_acpl:
            state["champion"] = {
                "file": f"mutations/{best_config.get('id')}.json",
                "fitness": best_candidate.acpl,
                "generation_discovered": current_gen
            }
            print(f"\n🏆 NEW CHAMPION: {best_config.get('id')} with ACPL {best_candidate.acpl:.1f}")
        else:
            print(f"\n📈 Best this gen: {best_config.get('id')} with ACPL {best_candidate.acpl:.1f}")
    else:
        print("\n⚠️  No valid candidates this generation!")

    # Update history
    best_acpl = min(c.acpl for c in candidates if c.is_valid) if valid_candidates else float("inf")
    state["history"].append({
        "generation": current_gen,
        "timestamp": datetime.now().isoformat(),
        "population_size": len(candidates),
        "best_acpl": best_acpl,
        "best_id": best_candidate.id if valid_candidates else None,
        "valid_count": sum(1 for c in candidates if c.is_valid)
    })

    # Phase 2: Selection & Reproduction
    print("\n🧬 Phase 2: Selection & Reproduction...")

    if valid_candidates and current_gen < state["config"]["max_generations"]:
        sorted_candidates = sorted(valid_candidates, key=lambda x: x[0].acpl)
        elite_count = state["config"].get("elite_count", 2)

        new_population = []

        # Keep elites
        for i, (cand, cfg) in enumerate(sorted_candidates[:elite_count]):
            print(f"   Elite {i+1}: {cfg.get('id')} (ACPL={cand.acpl:.1f})")
            new_population.append({
                "file": f"mutations/{cfg.get('id')}.json",
                "approach": cfg.get("description", "Elite survivor"),
                "fitness": cand.acpl
            })

        # Generate mutations
        next_gen = current_gen + 1
        mutation_count = 0

        for i, (_, elite_cfg) in enumerate(sorted_candidates[:2]):
            # 2 mutations per top elite, 1 for second
            num_mutations = 2 if i == 0 else 1
            for j in range(num_mutations):
                mutation_id = f"gen{next_gen}_mut{chr(ord('a') + mutation_count)}"
                mutated = mutate_config(elite_cfg, mutation_id)
                save_candidate_config(mutated, f"mutations/{mutation_id}.json")

                new_population.append({
                    "file": f"mutations/{mutation_id}.json",
                    "approach": mutated.get("description", "Mutation"),
                    "fitness": None
                })
                print(f"   Created mutation: {mutation_id}")
                mutation_count += 1

        # Generate crossover
        if len(sorted_candidates) >= 2:
            crossover_id = f"gen{next_gen}_cross"
            parent1_cfg = sorted_candidates[0][1]
            parent2_cfg = sorted_candidates[1][1]
            crossed = crossover_configs(parent1_cfg, parent2_cfg, crossover_id)
            save_candidate_config(crossed, f"mutations/{crossover_id}.json")

            new_population.append({
                "file": f"mutations/{crossover_id}.json",
                "approach": crossed.get("description", "Crossover"),
                "fitness": None
            })
            print(f"   Created crossover: {crossover_id}")

        state["population"] = new_population
        state["generation"] = next_gen

    # Phase 3: Documentation
    print("\n📝 Phase 3: Generating documentation...")

    # Analyze results
    learnings = analyze_results(candidates, configs)

    # Generate visualizations
    fitness_path = VISUALIZATIONS_DIR / f"fitness_gen{current_gen}.svg"
    generate_fitness_bar_chart(candidates, current_gen, str(fitness_path))
    print(f"   Generated: {fitness_path}")

    trajectory_path = VISUALIZATIONS_DIR / "evolution_trajectory.svg"
    generate_trajectory_chart(state["history"], str(trajectory_path))
    print(f"   Generated: {trajectory_path}")

    # Generate results markdown
    results_path = RESULTS_DIR / f"GEN{current_gen}_RESULTS.md"
    generate_results_markdown(
        current_gen,
        candidates,
        state["history"],
        learnings,
        str(results_path)
    )
    print(f"   Generated: {results_path}")

    # Save state
    save_evolution_state(state)
    print(f"\n✅ Generation {current_gen} complete!")
    print(f"   Next generation: {state['generation']}")
    print(f"   Champion ACPL: {state['champion']['fitness']:.1f}" if state.get('champion') else "   No champion yet")

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evolution generation")
    parser.add_argument("--generation", "-g", type=int, help="Specific generation to run")
    args = parser.parse_args()

    run_generation(args.generation)
