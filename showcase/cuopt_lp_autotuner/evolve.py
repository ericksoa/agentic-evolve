"""
evolve.py - Genetic algorithm for policy evolution.

Implements:
- Population-based evolution with tournament selection
- Mutation and crossover operators
- Multi-objective fitness with constraints
- Elitism and diversity preservation
- Checkpointing and resume
- Co-evolution with adversarial instances
"""

import json
import time
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import copy

from .policy import Policy, mutate_policy, crossover_policies, random_policy, BASELINE_POLICY
from .benchmark import Benchmark, FitnessResult, BenchmarkMetrics
from .lp_generator import create_corpus, LPGenerator, GeneratorConfig
from .transforms import LPInstance

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """An individual in the population."""
    policy: Policy
    fitness: Optional[FitnessResult] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.policy.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'policy': self.policy.to_dict(),
            'fitness': self.fitness.to_dict() if self.fitness else None,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Individual':
        policy = Policy.from_dict(d['policy'])
        fitness = None
        if d.get('fitness'):
            # Reconstruct FitnessResult (simplified)
            fitness = FitnessResult(
                primary_fitness=d['fitness']['primary_fitness'],
                throughput_score=d['fitness']['throughput_score'],
                latency_score=d['fitness']['latency_score'],
            )
        return cls(
            policy=policy,
            fitness=fitness,
            generation=d.get('generation', 0),
            parent_ids=d.get('parent_ids', []),
        )


@dataclass
class EvolutionState:
    """State of the evolutionary process."""
    generation: int = 0
    population: List[Individual] = field(default_factory=list)
    champion: Optional[Individual] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    stagnation_counter: int = 0
    best_fitness_ever: float = float('-inf')

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'champion': self.champion.to_dict() if self.champion else None,
            'history': self.history,
            'stagnation_counter': self.stagnation_counter,
            'best_fitness_ever': self.best_fitness_ever,
            'config': self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvolutionState':
        population = [Individual.from_dict(ind) for ind in d.get('population', [])]
        champion = Individual.from_dict(d['champion']) if d.get('champion') else None
        return cls(
            generation=d.get('generation', 0),
            population=population,
            champion=champion,
            history=d.get('history', []),
            stagnation_counter=d.get('stagnation_counter', 0),
            best_fitness_ever=d.get('best_fitness_ever', float('-inf')),
            config=d.get('config', {}),
        )

    def save(self, path: Path):
        """Save state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'EvolutionState':
        """Load state from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    # Population
    population_size: int = 20
    elite_size: int = 2
    tournament_size: int = 3

    # Operators
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7

    # Stopping criteria
    max_generations: int = 50
    stagnation_limit: int = 10
    target_fitness: Optional[float] = None

    # Corpus
    n_train: int = 100
    n_valid: int = 20
    n_test: int = 30
    corpus_seed: int = 42

    # Adversarial co-evolution
    adversarial_enabled: bool = True
    adversarial_every_n: int = 5
    adversarial_instances_per_gen: int = 5

    # Misc
    seed: int = 42
    output_dir: str = '.evolve-sdk/cuopt_lp_autotuner'
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evolution:
    """
    Genetic algorithm for evolving LP solver policies.

    Usage:
        evo = Evolution(config)
        evo.run()
        best = evo.get_champion()
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.rng = random.Random(self.config.seed)

        # Initialize state
        self.state = EvolutionState(config=self.config.to_dict())

        # Create corpus
        self.corpus = create_corpus(
            n_train=self.config.n_train,
            n_valid=self.config.n_valid,
            n_test=self.config.n_test,
            seed=self.config.corpus_seed,
        )

        # Create benchmark
        self.benchmark = Benchmark(
            self.corpus,
            verbose=self.config.verbose,
        )

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Individual:
        """
        Run the evolution process.

        Returns:
            Champion individual
        """
        logger.info("Starting evolution")

        # Initialize population if empty
        if not self.state.population:
            self._initialize_population()

        # Evolution loop
        while not self._should_stop():
            self.state.generation += 1

            if self.config.verbose:
                print(f"\n=== Generation {self.state.generation} ===")

            # Evaluate population
            self._evaluate_population()

            # Update champion
            self._update_champion()

            # Record history
            self._record_history()

            # Check stagnation
            self._check_stagnation()

            # Adversarial co-evolution
            if (self.config.adversarial_enabled and
                self.state.generation % self.config.adversarial_every_n == 0):
                self._add_adversarial_instances()

            # Create next generation
            self._evolve_population()

            # Save checkpoint
            self._save_checkpoint()

        # Final evaluation on test set
        self._final_evaluation()

        return self.state.champion

    def _initialize_population(self):
        """Initialize the population with diverse individuals."""
        logger.info(f"Initializing population of size {self.config.population_size}")

        # Always include baseline
        baseline = Individual(
            policy=Policy.baseline(),
            generation=0,
        )
        baseline.policy.name = "baseline"
        self.state.population.append(baseline)

        # Generate random individuals
        for i in range(self.config.population_size - 1):
            policy = random_policy(self.rng)
            policy.name = f"random_{i:03d}"
            individual = Individual(policy=policy, generation=0)
            self.state.population.append(individual)

    def _evaluate_population(self):
        """Evaluate fitness for all unevaluated individuals."""
        for ind in self.state.population:
            if ind.fitness is None:
                try:
                    ind.fitness = self.benchmark.compute_fitness(ind.policy, 'train')
                    if self.config.verbose:
                        print(f"  {ind.id}: fitness={ind.fitness.primary_fitness:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate {ind.id}: {e}")
                    ind.fitness = FitnessResult(primary_fitness=-9999)

    def _update_champion(self):
        """Update the champion if a better individual is found."""
        valid_individuals = [ind for ind in self.state.population if ind.fitness is not None]

        if not valid_individuals:
            return

        best = max(valid_individuals, key=lambda x: x.fitness.primary_fitness)

        if (self.state.champion is None or
            best.fitness.primary_fitness > self.state.best_fitness_ever):

            self.state.champion = copy.deepcopy(best)
            self.state.best_fitness_ever = best.fitness.primary_fitness
            self.state.stagnation_counter = 0

            if self.config.verbose:
                print(f"  New champion: {best.id} with fitness {best.fitness.primary_fitness:.2f}")

            # Save champion policy
            champion_path = self.output_dir / 'best_policy.json'
            self.state.champion.policy.to_json(champion_path)

    def _record_history(self):
        """Record generation statistics."""
        fitnesses = [ind.fitness.primary_fitness for ind in self.state.population
                     if ind.fitness is not None]

        if not fitnesses:
            return

        record = {
            'generation': self.state.generation,
            'best_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'champion_id': self.state.champion.id if self.state.champion else None,
            'timestamp': time.time(),
        }

        self.state.history.append(record)

        if self.config.verbose:
            print(f"  Best: {record['best_fitness']:.2f}, Mean: {record['mean_fitness']:.2f}")

    def _check_stagnation(self):
        """Check for stagnation and potentially restart."""
        if len(self.state.history) < 2:
            return

        current_best = self.state.history[-1]['best_fitness']
        previous_best = self.state.history[-2]['best_fitness']

        if current_best <= previous_best:
            self.state.stagnation_counter += 1
        else:
            self.state.stagnation_counter = 0

        if self.state.stagnation_counter >= self.config.stagnation_limit:
            if self.config.verbose:
                print(f"  Stagnation detected ({self.state.stagnation_counter} generations)")

    def _evolve_population(self):
        """Create the next generation through selection, crossover, and mutation."""
        # Sort by fitness
        sorted_pop = sorted(
            [ind for ind in self.state.population if ind.fitness is not None],
            key=lambda x: x.fitness.primary_fitness,
            reverse=True
        )

        new_population = []

        # Elitism: keep best individuals
        for i in range(min(self.config.elite_size, len(sorted_pop))):
            elite = copy.deepcopy(sorted_pop[i])
            elite.policy.name = f"elite_{i}_gen{self.state.generation}"
            new_population.append(elite)

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            if self.rng.random() < self.config.crossover_rate and len(sorted_pop) >= 2:
                # Crossover
                parent1 = self._tournament_select(sorted_pop)
                parent2 = self._tournament_select(sorted_pop)
                child_policy = crossover_policies(parent1.policy, parent2.policy, self.rng)
                child_policy.name = f"cross_{len(new_population)}_gen{self.state.generation}"
                child = Individual(
                    policy=child_policy,
                    generation=self.state.generation,
                    parent_ids=[parent1.id, parent2.id],
                )
            else:
                # Mutation only
                parent = self._tournament_select(sorted_pop)
                child_policy = mutate_policy(parent.policy, self.config.mutation_rate, self.rng)
                child_policy.name = f"mut_{len(new_population)}_gen{self.state.generation}"
                child = Individual(
                    policy=child_policy,
                    generation=self.state.generation,
                    parent_ids=[parent.id],
                )

            # Apply additional mutation
            if self.rng.random() < self.config.mutation_rate:
                child.policy = mutate_policy(child.policy, self.config.mutation_rate * 0.5, self.rng)

            new_population.append(child)

        self.state.population = new_population

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Select an individual using tournament selection."""
        tournament = self.rng.sample(
            population,
            min(self.config.tournament_size, len(population))
        )
        return max(tournament, key=lambda x: x.fitness.primary_fitness)

    def _add_adversarial_instances(self):
        """Add adversarial instances to the training corpus."""
        if self.state.champion is None:
            return

        logger.info("Generating adversarial instances...")

        from .adversarial import AdversarialGenerator

        adv_gen = AdversarialGenerator(
            target_policy=self.state.champion.policy,
            benchmark=self.benchmark,
            seed=self.config.seed + self.state.generation,
        )

        new_instances = adv_gen.generate(self.config.adversarial_instances_per_gen)

        if new_instances:
            self.corpus['train'].extend(new_instances)
            # Rebuild benchmark with updated corpus
            self.benchmark = Benchmark(self.corpus, verbose=self.config.verbose)

            if self.config.verbose:
                print(f"  Added {len(new_instances)} adversarial instances")

    def _should_stop(self) -> bool:
        """Check if evolution should stop."""
        if self.state.generation >= self.config.max_generations:
            logger.info("Reached max generations")
            return True

        if (self.config.target_fitness is not None and
            self.state.best_fitness_ever >= self.config.target_fitness):
            logger.info("Reached target fitness")
            return True

        if self.state.stagnation_counter >= self.config.stagnation_limit * 2:
            logger.info("Extended stagnation - stopping")
            return True

        return False

    def _save_checkpoint(self):
        """Save evolution state checkpoint."""
        checkpoint_path = self.output_dir / 'evolution.json'
        self.state.save(checkpoint_path)

    def _final_evaluation(self):
        """Evaluate champion on test set."""
        if self.state.champion is None:
            return

        if self.config.verbose:
            print("\n=== Final Evaluation on Test Set ===")

        test_fitness = self.benchmark.compute_fitness(self.state.champion.policy, 'test')

        print(f"Champion: {self.state.champion.id}")
        print(f"  Train fitness: {self.state.champion.fitness.primary_fitness:.2f}")
        print(f"  Test fitness: {test_fitness.primary_fitness:.2f}")
        print(f"  Throughput: {test_fitness.metrics.throughput_lps_per_sec:.2f} LPs/sec")
        print(f"  p95 latency: {test_fitness.metrics.latency_p95_ms:.2f} ms")
        print(f"  Success rate: {test_fitness.metrics.success_rate:.1%}")

        if test_fitness.baseline_metrics:
            baseline = test_fitness.baseline_metrics
            print(f"\nBaseline:")
            print(f"  Throughput: {baseline.throughput_lps_per_sec:.2f} LPs/sec")
            print(f"  p95 latency: {baseline.latency_p95_ms:.2f} ms")

            throughput_improvement = (
                (test_fitness.metrics.throughput_lps_per_sec / baseline.throughput_lps_per_sec - 1) * 100
                if baseline.throughput_lps_per_sec > 0 else 0
            )
            latency_improvement = (
                (1 - test_fitness.metrics.latency_p95_ms / baseline.latency_p95_ms) * 100
                if baseline.latency_p95_ms > 0 else 0
            )

            print(f"\nImprovement:")
            print(f"  Throughput: {throughput_improvement:+.1f}%")
            print(f"  p95 latency: {latency_improvement:+.1f}%")

        # Save final results
        results = {
            'champion': self.state.champion.to_dict(),
            'train_fitness': self.state.champion.fitness.to_dict(),
            'test_fitness': test_fitness.to_dict(),
            'generations': self.state.generation,
            'history': self.state.history,
        }

        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def get_champion(self) -> Optional[Individual]:
        """Get the current champion."""
        return self.state.champion

    @classmethod
    def resume(cls, output_dir: str) -> 'Evolution':
        """Resume evolution from a checkpoint."""
        output_path = Path(output_dir)
        checkpoint_path = output_path / 'evolution.json'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        state = EvolutionState.load(checkpoint_path)
        config = EvolutionConfig(**state.config)

        evo = cls(config)
        evo.state = state

        logger.info(f"Resumed from generation {state.generation}")
        return evo


def run_evolution(
    config: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    output_dir: str = '.evolve-sdk/cuopt_lp_autotuner',
) -> Dict[str, Any]:
    """
    Main entry point for running evolution.

    Args:
        config: Evolution configuration dict
        resume: Whether to resume from checkpoint
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    if resume:
        evo = Evolution.resume(output_dir)
    else:
        evo_config = EvolutionConfig(**(config or {}))
        evo_config.output_dir = output_dir
        evo = Evolution(evo_config)

    champion = evo.run()

    return {
        'champion_id': champion.id if champion else None,
        'champion_fitness': champion.fitness.primary_fitness if champion and champion.fitness else None,
        'generations': evo.state.generation,
        'output_dir': output_dir,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evolve LP solver policies')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--generations', type=int, default=20, help='Max generations')
    parser.add_argument('--population', type=int, default=15, help='Population size')
    parser.add_argument('--output', type=str, default='.evolve-sdk/cuopt_lp_autotuner',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    config = {
        'max_generations': args.generations,
        'population_size': args.population,
        'seed': args.seed,
        'verbose': not args.quiet,
        'output_dir': args.output,
    }

    results = run_evolution(config=config, resume=args.resume, output_dir=args.output)

    print(f"\nEvolution complete!")
    print(f"  Champion: {results['champion_id']}")
    print(f"  Fitness: {results['champion_fitness']:.2f}")
    print(f"  Generations: {results['generations']}")
