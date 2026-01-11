#!/usr/bin/env python3
"""
Evolutionary Prompt Optimization for Chess LLM

Evolves prompt configurations to minimize ACPL (Average Centipawn Loss).

Key features:
- Population-based evolution with elitism
- Semantic crossover (combine best traits from parents)
- Targeted mutations (vary one dimension at a time)
- Adaptive stopping (plateau detection)

Usage:
    python evolve_prompts.py --generations 10 --population 6
    python evolve_prompts.py --resume
"""
import sys
import json
import random
import copy
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from prompt_template import (
    PromptConfig, FewShotExample, BASELINE,
    SYSTEM_PROMPT_VARIANTS, USER_TEMPLATE_VARIANTS,
    FORMAT_HINT_VARIANTS, CANDIDATE_FEW_SHOTS
)
from fitness_evaluator import evaluate_fitness, FitnessResult


@dataclass
class Individual:
    """An individual in the population"""
    id: str
    generation: int
    config: PromptConfig
    fitness: Optional[FitnessResult] = None
    parent_ids: List[str] = None
    mutation_type: str = "initial"

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


@dataclass
class EvolutionState:
    """Full evolution state for checkpointing"""
    generation: int
    population: List[Individual]
    champion: Optional[Individual]
    history: List[dict]  # Per-generation summaries
    baseline_fitness: float
    start_time: str
    last_update: str

    def save(self, path: str):
        """Save state to JSON"""
        data = {
            "generation": self.generation,
            "population": [
                {
                    "id": ind.id,
                    "generation": ind.generation,
                    "config": ind.config.to_dict(),
                    "fitness": ind.fitness.to_dict() if ind.fitness else None,
                    "parent_ids": ind.parent_ids,
                    "mutation_type": ind.mutation_type,
                }
                for ind in self.population
            ],
            "champion": {
                "id": self.champion.id,
                "generation": self.champion.generation,
                "config": self.champion.config.to_dict(),
                "fitness": self.champion.fitness.to_dict() if self.champion.fitness else None,
                "parent_ids": self.champion.parent_ids,
                "mutation_type": self.champion.mutation_type,
            } if self.champion else None,
            "history": self.history,
            "baseline_fitness": self.baseline_fitness,
            "start_time": self.start_time,
            "last_update": self.last_update,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvolutionState":
        """Load state from JSON"""
        with open(path) as f:
            data = json.load(f)

        def dict_to_individual(d):
            config = PromptConfig.from_dict(d["config"])
            fitness = FitnessResult(**d["fitness"]) if d["fitness"] else None
            return Individual(
                id=d["id"],
                generation=d["generation"],
                config=config,
                fitness=fitness,
                parent_ids=d.get("parent_ids", []),
                mutation_type=d.get("mutation_type", "unknown"),
            )

        population = [dict_to_individual(d) for d in data["population"]]
        champion = dict_to_individual(data["champion"]) if data["champion"] else None

        return cls(
            generation=data["generation"],
            population=population,
            champion=champion,
            history=data["history"],
            baseline_fitness=data["baseline_fitness"],
            start_time=data["start_time"],
            last_update=data["last_update"],
        )


def generate_id() -> str:
    """Generate unique ID"""
    import uuid
    return uuid.uuid4().hex[:8]


def mutate_system_prompt(config: PromptConfig) -> PromptConfig:
    """Mutate system prompt"""
    new_config = copy.deepcopy(config)
    # Pick a different variant
    options = [p for p in SYSTEM_PROMPT_VARIANTS if p != config.system_prompt]
    if options:
        new_config.system_prompt = random.choice(options)
    return new_config


def mutate_user_template(config: PromptConfig) -> PromptConfig:
    """Mutate user template"""
    new_config = copy.deepcopy(config)
    options = [t for t in USER_TEMPLATE_VARIANTS if t != config.user_template]
    if options:
        new_config.user_template = random.choice(options)
    return new_config


def mutate_format_hint(config: PromptConfig) -> PromptConfig:
    """Mutate format hint"""
    new_config = copy.deepcopy(config)
    options = [h for h in FORMAT_HINT_VARIANTS if h != config.format_hint]
    if options:
        new_config.format_hint = random.choice(options)
    return new_config


def mutate_few_shots(config: PromptConfig) -> PromptConfig:
    """Mutate few-shot examples"""
    new_config = copy.deepcopy(config)

    # Options: add, remove, or swap
    n_current = len(new_config.few_shot_examples)

    if n_current == 0:
        # Add 1-2 examples
        n_add = random.randint(1, 2)
        new_config.few_shot_examples = random.sample(CANDIDATE_FEW_SHOTS, min(n_add, len(CANDIDATE_FEW_SHOTS)))
    elif n_current >= 3:
        # Remove one
        new_config.few_shot_examples = random.sample(new_config.few_shot_examples, n_current - 1)
    else:
        # 50% add, 50% remove
        if random.random() < 0.5 and n_current < len(CANDIDATE_FEW_SHOTS):
            # Add one
            available = [ex for ex in CANDIDATE_FEW_SHOTS if ex not in new_config.few_shot_examples]
            if available:
                new_config.few_shot_examples.append(random.choice(available))
        else:
            # Remove one
            if n_current > 0:
                new_config.few_shot_examples = random.sample(new_config.few_shot_examples, n_current - 1)

    return new_config


def mutate_temperature(config: PromptConfig) -> PromptConfig:
    """Mutate temperature"""
    new_config = copy.deepcopy(config)
    # Small variations around current value
    temps = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    options = [t for t in temps if t != config.temperature]
    if options:
        new_config.temperature = random.choice(options)
    return new_config


MUTATION_FUNCTIONS = [
    ("system_prompt", mutate_system_prompt),
    ("user_template", mutate_user_template),
    ("format_hint", mutate_format_hint),
    ("few_shots", mutate_few_shots),
    ("temperature", mutate_temperature),
]


def mutate(parent: Individual, generation: int) -> Individual:
    """Create a mutated offspring"""
    # Pick random mutation type
    mutation_name, mutation_fn = random.choice(MUTATION_FUNCTIONS)

    new_config = mutation_fn(parent.config)

    return Individual(
        id=generate_id(),
        generation=generation,
        config=new_config,
        parent_ids=[parent.id],
        mutation_type=mutation_name,
    )


def crossover(parent1: Individual, parent2: Individual, generation: int) -> Individual:
    """Create offspring by combining two parents"""
    config1 = parent1.config
    config2 = parent2.config

    # Randomly pick each attribute from one parent
    new_config = PromptConfig(
        system_prompt=random.choice([config1.system_prompt, config2.system_prompt]),
        user_template=random.choice([config1.user_template, config2.user_template]),
        format_hint=random.choice([config1.format_hint, config2.format_hint]),
        few_shot_examples=random.choice([config1.few_shot_examples, config2.few_shot_examples]),
        temperature=random.choice([config1.temperature, config2.temperature]),
        force_prefix=config1.force_prefix,  # Keep this constant
    )

    return Individual(
        id=generate_id(),
        generation=generation,
        config=new_config,
        parent_ids=[parent1.id, parent2.id],
        mutation_type="crossover",
    )


def select_parents(population: List[Individual], n: int = 2) -> List[Individual]:
    """Tournament selection"""
    # Filter to evaluated individuals
    evaluated = [ind for ind in population if ind.fitness is not None]
    if len(evaluated) < n:
        return evaluated

    # Tournament: pick best from random subset
    selected = []
    for _ in range(n):
        tournament = random.sample(evaluated, min(3, len(evaluated)))
        winner = min(tournament, key=lambda x: x.fitness.acpl)
        selected.append(winner)

    return selected


def evolve(
    generations: int = 10,
    population_size: int = 6,
    positions_per_eval: int = 30,
    depth: int = 12,
    elite_count: int = 2,
    output_dir: Path = None,
    resume: bool = False,
):
    """
    Run evolutionary optimization.

    Args:
        generations: Max generations to run
        population_size: Number of individuals per generation
        positions_per_eval: Chess positions to evaluate each individual
        depth: Stockfish analysis depth
        elite_count: Number of top performers to keep each generation
        output_dir: Where to save state
        resume: Resume from checkpoint
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    state_path = output_dir / "evolution.json"
    mutations_dir = output_dir / "mutations"
    mutations_dir.mkdir(exist_ok=True)

    # Resume or initialize
    if resume and state_path.exists():
        print("Resuming from checkpoint...")
        state = EvolutionState.load(str(state_path))
        start_gen = state.generation + 1
    else:
        print("Initializing evolution...")

        # Evaluate baseline
        print("\nEvaluating baseline...")
        baseline_result = evaluate_fitness(BASELINE, n_positions=positions_per_eval, depth=depth)
        print(f"Baseline ACPL: {baseline_result.acpl:.1f}")

        baseline_ind = Individual(
            id="baseline",
            generation=0,
            config=BASELINE,
            fitness=baseline_result,
            mutation_type="baseline",
        )

        state = EvolutionState(
            generation=0,
            population=[baseline_ind],
            champion=baseline_ind,
            history=[{
                "generation": 0,
                "best_acpl": baseline_result.acpl,
                "avg_acpl": baseline_result.acpl,
                "best_id": "baseline",
                "population_size": 1,
            }],
            baseline_fitness=baseline_result.acpl,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
        )

        state.save(str(state_path))
        start_gen = 1

    print(f"\nStarting evolution from generation {start_gen}")
    print(f"Baseline ACPL: {state.baseline_fitness:.1f}")
    print(f"Current champion ACPL: {state.champion.fitness.acpl:.1f}")
    print()

    # Evolution loop
    for gen in range(start_gen, generations + 1):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}")
        print(f"{'='*60}")

        new_population = []

        # Elitism: keep top performers
        evaluated = [ind for ind in state.population if ind.fitness is not None]
        evaluated.sort(key=lambda x: x.fitness.acpl)
        elites = evaluated[:elite_count]
        new_population.extend(elites)
        print(f"Kept {len(elites)} elites (best ACPL: {elites[0].fitness.acpl:.1f})")

        # Generate offspring
        offspring_needed = population_size - len(new_population)
        offspring = []

        for i in range(offspring_needed):
            # 70% mutation, 30% crossover
            if random.random() < 0.7 or len(evaluated) < 2:
                # Mutation
                parent = select_parents(evaluated, n=1)[0]
                child = mutate(parent, gen)
            else:
                # Crossover
                parents = select_parents(evaluated, n=2)
                child = crossover(parents[0], parents[1], gen)

            offspring.append(child)

        # Evaluate offspring
        print(f"\nEvaluating {len(offspring)} offspring...")
        for i, child in enumerate(offspring):
            print(f"\n  [{i+1}/{len(offspring)}] {child.mutation_type} (from {child.parent_ids})")
            print(f"      System: {child.config.system_prompt[:40]}...")
            print(f"      Few-shots: {len(child.config.few_shot_examples)}")

            try:
                fitness = evaluate_fitness(child.config, n_positions=positions_per_eval, depth=depth, verbose=False)
                child.fitness = fitness
                print(f"      ACPL: {fitness.acpl:.1f} (legal: {fitness.legal_rate*100:.0f}%)")

                # Save config
                child.config.save(str(mutations_dir / f"gen{gen}_{child.id}.json"))
            except Exception as e:
                print(f"      ERROR: {e}")
                # Assign worst fitness
                child.fitness = FitnessResult(
                    acpl=10000, acpl_legal_only=10000, legal_rate=0,
                    best_move_rate=0, n_positions=0, n_legal=0, n_illegal=0
                )

        new_population.extend(offspring)

        # Update champion
        all_evaluated = [ind for ind in new_population if ind.fitness is not None]
        if all_evaluated:
            gen_best = min(all_evaluated, key=lambda x: x.fitness.acpl)
            if gen_best.fitness.acpl < state.champion.fitness.acpl:
                print(f"\n*** NEW CHAMPION: {gen_best.id} ***")
                print(f"    ACPL: {gen_best.fitness.acpl:.1f} (was {state.champion.fitness.acpl:.1f})")
                state.champion = gen_best

        # Update state
        state.population = new_population
        state.generation = gen
        state.last_update = datetime.now().isoformat()

        # Record history
        acpls = [ind.fitness.acpl for ind in all_evaluated]
        state.history.append({
            "generation": gen,
            "best_acpl": min(acpls) if acpls else 10000,
            "avg_acpl": sum(acpls) / len(acpls) if acpls else 10000,
            "best_id": gen_best.id if all_evaluated else None,
            "population_size": len(new_population),
        })

        # Save checkpoint
        state.save(str(state_path))

        # Print generation summary
        print(f"\nGeneration {gen} Summary:")
        print(f"  Best ACPL: {min(acpls):.1f}")
        print(f"  Avg ACPL:  {sum(acpls)/len(acpls):.1f}")
        print(f"  Champion:  {state.champion.id} (ACPL: {state.champion.fitness.acpl:.1f})")

        # Improvement from baseline
        improvement = state.baseline_fitness - state.champion.fitness.acpl
        print(f"  Improvement from baseline: {improvement:.1f} ({improvement/state.baseline_fitness*100:.1f}%)")

    # Final summary
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"\nBaseline ACPL:  {state.baseline_fitness:.1f}")
    print(f"Champion ACPL:  {state.champion.fitness.acpl:.1f}")
    print(f"Improvement:    {state.baseline_fitness - state.champion.fitness.acpl:.1f}")
    print(f"\nChampion config:")
    print(f"  System: {state.champion.config.system_prompt}")
    print(f"  Template: {state.champion.config.user_template}")
    print(f"  Few-shots: {len(state.champion.config.few_shot_examples)}")
    print(f"  Temperature: {state.champion.config.temperature}")

    # Save champion
    champion_path = output_dir / "champion.json"
    state.champion.config.save(str(champion_path))
    print(f"\nChampion saved to: {champion_path}")

    return state


def main():
    parser = argparse.ArgumentParser(description="Evolve chess prompts")
    parser.add_argument("--generations", type=int, default=8, help="Max generations")
    parser.add_argument("--population", type=int, default=5, help="Population size")
    parser.add_argument("--positions", type=int, default=25, help="Positions per evaluation")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish depth")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    evolve(
        generations=args.generations,
        population_size=args.population,
        positions_per_eval=args.positions,
        depth=args.depth,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
