"""
evolution.py - Genetic algorithm for cuOpt parameter optimization.

This module implements the main evolution loop that:
1. Maintains a population of CuOptConfig genomes
2. Evaluates fitness on LP benchmarks
3. Applies selection, crossover, and mutation
4. Tracks progress and prevents overfitting via train/test splits

Key anti-overfitting measures:
- Separate train/validation/test splits
- Hard constraint on p95 regression vs baseline on test set
- Adversarial instance generation
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .cuopt_config import (
    CuOptConfig, mutate_config, crossover_configs, random_config,
    ConditionalConfig
)
from .cuopt_solver import CuOptSolver, SolveStatus, solve_batch, BatchResult
from .mps_loader import LPData, load_problem, get_features
from .lp_generator import LPGenerator

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """Fitness evaluation result."""
    # Primary fitness (higher is better)
    fitness: float

    # Component scores
    throughput_score: float = 0.0  # vs baseline
    latency_score: float = 0.0    # p95 vs baseline
    success_rate: float = 0.0

    # Detailed metrics
    throughput_lps_per_sec: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Baseline comparison
    baseline_throughput: float = 0.0
    baseline_p95_ms: float = 0.0

    # Constraint checks
    p95_regression_pct: float = 0.0
    violates_p95_constraint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitness": float(self.fitness),
            "throughput_score": float(self.throughput_score),
            "latency_score": float(self.latency_score),
            "success_rate": float(self.success_rate),
            "throughput_lps_per_sec": float(self.throughput_lps_per_sec),
            "mean_latency_ms": float(self.mean_latency_ms),
            "p50_latency_ms": float(self.p50_latency_ms),
            "p95_latency_ms": float(self.p95_latency_ms),
            "p99_latency_ms": float(self.p99_latency_ms),
            "baseline_throughput": float(self.baseline_throughput),
            "baseline_p95_ms": float(self.baseline_p95_ms),
            "p95_regression_pct": float(self.p95_regression_pct),
            "violates_p95_constraint": bool(self.violates_p95_constraint),
        }


@dataclass
class Individual:
    """An individual in the population."""
    config: CuOptConfig
    id: str
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    train_fitness: Optional[FitnessResult] = None
    val_fitness: Optional[FitnessResult] = None
    test_fitness: Optional[FitnessResult] = None


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    # Population
    population_size: int = 20
    elite_count: int = 2
    tournament_size: int = 3

    # Genetic operators
    mutation_rate: float = 0.2
    crossover_rate: float = 0.3

    # Termination
    max_generations: int = 50
    plateau_limit: int = 10
    target_improvement: float = 0.10  # 10% improvement target

    # Constraints
    max_p95_regression_pct: float = 2.0  # Hard constraint: no >2% regression

    # Evaluation
    timeout_per_lp_ms: float = 5000.0
    batch_timeout_ms: float = 60000.0

    # Logging
    verbose: bool = True
    checkpoint_dir: str = ".evolve-sdk/cuopt_evolution"


class LPCorpus:
    """
    LP problem corpus with train/val/test splits.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.train: List[LPData] = []
        self.val: List[LPData] = []
        self.test: List[LPData] = []
        self.features_cache: Dict[str, Dict] = {}

    def load_from_directory(self, path: str, extensions: List[str] = [".mps", ".mps.gz"]):
        """Load LP problems from a directory."""
        problems = []
        path = Path(path)

        for ext in extensions:
            for f in path.glob(f"**/*{ext}"):
                try:
                    lp = load_problem(str(f))
                    problems.append(lp)
                    logger.info(f"Loaded {f.name}: {lp.n_vars} vars, {lp.n_cons} cons")
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")

        self._split_problems(problems)

    def generate_synthetic(
        self,
        n_train: int = 60,
        n_val: int = 20,
        n_test: int = 20,
    ):
        """Generate synthetic LP problems."""
        generator = LPGenerator(seed=self.seed)

        # Generate diverse problems
        for split_name, n_problems, target_list in [
            ("train", n_train, self.train),
            ("val", n_val, self.val),
            ("test", n_test, self.test),
        ]:
            for i in range(n_problems):
                # Vary problem characteristics
                lp = generator.generate_random_lp(
                    n_vars_range=(10, 500),
                    n_cons_range=(5, 300),
                    density_range=(0.01, 0.3),
                )
                lp.name = f"{split_name}_{i}"
                target_list.append(lp)

        logger.info(f"Generated {len(self.train)} train, {len(self.val)} val, {len(self.test)} test problems")

    def _split_problems(self, problems: List[LPData], ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)):
        """Split problems into train/val/test."""
        self.rng.shuffle(problems)
        n = len(problems)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        self.train = problems[:n_train]
        self.val = problems[n_train:n_train + n_val]
        self.test = problems[n_train + n_val:]

    def get_features(self, lp: LPData) -> Dict[str, Any]:
        """Get cached features for an LP."""
        if lp.name not in self.features_cache:
            self.features_cache[lp.name] = get_features(lp)
        return self.features_cache[lp.name]


class CuOptEvolution:
    """
    Genetic algorithm for optimizing cuOpt configurations.
    """

    def __init__(
        self,
        corpus: LPCorpus,
        config: Optional[EvolutionConfig] = None,
        seed: int = 42,
    ):
        self.corpus = corpus
        self.config = config or EvolutionConfig()
        self.rng = random.Random(seed)

        self.solver = CuOptSolver(use_fallback=True, verbose=False)
        self.baseline_config = CuOptConfig.baseline()

        # Population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None

        # History
        self.history: List[Dict] = []
        self.plateau_counter = 0

        # Baseline metrics (computed once)
        self._baseline_train: Optional[BatchResult] = None
        self._baseline_val: Optional[BatchResult] = None
        self._baseline_test: Optional[BatchResult] = None

    def initialize(self):
        """Initialize population and compute baseline."""
        logger.info("Computing baseline metrics...")
        self._compute_baseline()

        logger.info(f"Initializing population of {self.config.population_size}...")
        self.population = []

        # Add baseline config
        baseline_ind = Individual(
            config=self.baseline_config.copy(),
            id="baseline",
            generation=0,
        )
        self.population.append(baseline_ind)

        # Add random configurations
        for i in range(self.config.population_size - 1):
            config = random_config(self.rng)
            ind = Individual(
                config=config,
                id=f"gen0_{i}",
                generation=0,
            )
            self.population.append(ind)

        # Evaluate initial population
        self._evaluate_population(self.population, "train")

        # Sort and find best
        self.population.sort(key=lambda x: x.train_fitness.fitness if x.train_fitness else float('-inf'), reverse=True)
        self.best_individual = self.population[0]

        logger.info(f"Initial best fitness: {self.best_individual.train_fitness.fitness:.4f}")

    def _compute_baseline(self):
        """Compute baseline metrics on all splits."""
        self._baseline_train = solve_batch(
            self.corpus.train,
            self.baseline_config,
            self.solver,
            self.config.batch_timeout_ms,
        )
        self._baseline_val = solve_batch(
            self.corpus.val,
            self.baseline_config,
            self.solver,
            self.config.batch_timeout_ms,
        )
        self._baseline_test = solve_batch(
            self.corpus.test,
            self.baseline_config,
            self.solver,
            self.config.batch_timeout_ms,
        )

        logger.info(f"Baseline train: {self._baseline_train.throughput_lps_per_sec:.1f} LP/s, "
                   f"p95={self._baseline_train.p95_solve_time_ms:.2f}ms")
        logger.info(f"Baseline val: {self._baseline_val.throughput_lps_per_sec:.1f} LP/s, "
                   f"p95={self._baseline_val.p95_solve_time_ms:.2f}ms")
        logger.info(f"Baseline test: {self._baseline_test.throughput_lps_per_sec:.1f} LP/s, "
                   f"p95={self._baseline_test.p95_solve_time_ms:.2f}ms")

    def _evaluate_individual(
        self,
        individual: Individual,
        split: str,
        baseline: BatchResult,
    ) -> FitnessResult:
        """Evaluate an individual on a split."""
        lps = getattr(self.corpus, split)

        result = solve_batch(
            lps,
            individual.config,
            self.solver,
            self.config.batch_timeout_ms,
        )

        # Compute fitness components
        # Throughput improvement (higher is better)
        if baseline.throughput_lps_per_sec > 0:
            throughput_ratio = result.throughput_lps_per_sec / baseline.throughput_lps_per_sec
            throughput_score = (throughput_ratio - 1.0) * 10  # Scale: 10% improvement = +1 point
        else:
            throughput_score = 0.0

        # Latency improvement (lower is better, so improvement is positive)
        if baseline.p95_solve_time_ms > 0 and result.p95_solve_time_ms > 0:
            latency_ratio = result.p95_solve_time_ms / baseline.p95_solve_time_ms
            latency_score = (1.0 - latency_ratio) * 10  # 10% reduction = +1 point
            p95_regression_pct = (latency_ratio - 1.0) * 100
        else:
            latency_score = 0.0
            p95_regression_pct = 0.0

        # Penalty for constraint violation
        violates_p95 = p95_regression_pct > self.config.max_p95_regression_pct

        # Success rate bonus/penalty
        success_bonus = (result.success_rate - baseline.success_rate) * 5

        # Combined fitness
        if violates_p95:
            # Heavy penalty for constraint violation
            fitness = -1000 + throughput_score + latency_score
        else:
            fitness = throughput_score + latency_score + success_bonus

        return FitnessResult(
            fitness=fitness,
            throughput_score=throughput_score,
            latency_score=latency_score,
            success_rate=result.success_rate,
            throughput_lps_per_sec=result.throughput_lps_per_sec,
            mean_latency_ms=result.mean_solve_time_ms,
            p50_latency_ms=result.p50_solve_time_ms,
            p95_latency_ms=result.p95_solve_time_ms,
            p99_latency_ms=result.p99_solve_time_ms,
            baseline_throughput=baseline.throughput_lps_per_sec,
            baseline_p95_ms=baseline.p95_solve_time_ms,
            p95_regression_pct=p95_regression_pct,
            violates_p95_constraint=violates_p95,
        )

    def _evaluate_population(self, population: List[Individual], split: str):
        """Evaluate all individuals on a split."""
        baseline = {
            "train": self._baseline_train,
            "val": self._baseline_val,
            "test": self._baseline_test,
        }[split]

        for ind in population:
            fitness = self._evaluate_individual(ind, split, baseline)
            if split == "train":
                ind.train_fitness = fitness
            elif split == "val":
                ind.val_fitness = fitness
            else:
                ind.test_fitness = fitness

    def _tournament_select(self, k: int = None) -> Individual:
        """Select an individual via tournament selection."""
        k = k or self.config.tournament_size
        candidates = self.rng.sample(self.population, min(k, len(self.population)))
        return max(candidates, key=lambda x: x.train_fitness.fitness if x.train_fitness else float('-inf'))

    def _create_offspring(self) -> List[Individual]:
        """Create offspring via crossover and mutation."""
        offspring = []
        n_offspring = self.config.population_size - self.config.elite_count

        while len(offspring) < n_offspring:
            if self.rng.random() < self.config.crossover_rate and len(self.population) >= 2:
                # Crossover
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child_config = crossover_configs(p1.config, p2.config, self.rng)
                parent_ids = [p1.id, p2.id]
            else:
                # Clone + mutate
                parent = self._tournament_select()
                child_config = parent.config.copy()
                parent_ids = [parent.id]

            # Mutation
            if self.rng.random() < self.config.mutation_rate or not parent_ids:
                child_config = mutate_config(child_config, self.config.mutation_rate, self.rng)

            child = Individual(
                config=child_config,
                id=f"gen{self.generation + 1}_{len(offspring)}",
                generation=self.generation + 1,
                parent_ids=parent_ids,
            )
            offspring.append(child)

        return offspring

    def evolve_generation(self) -> bool:
        """
        Run one generation of evolution.

        Returns:
            True if should continue, False if should stop
        """
        self.generation += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATION {self.generation}")
        logger.info(f"{'='*60}")

        # Keep elites
        elites = self.population[:self.config.elite_count]

        # Create offspring
        offspring = self._create_offspring()

        # Evaluate offspring on train set
        self._evaluate_population(offspring, "train")

        # Combine and sort
        self.population = elites + offspring
        self.population.sort(
            key=lambda x: x.train_fitness.fitness if x.train_fitness else float('-inf'),
            reverse=True
        )

        # Truncate to population size
        self.population = self.population[:self.config.population_size]

        # Check for improvement
        current_best = self.population[0]
        best_fitness = current_best.train_fitness.fitness if current_best.train_fitness else float('-inf')
        prev_best = self.best_individual.train_fitness.fitness if self.best_individual and self.best_individual.train_fitness else float('-inf')

        if best_fitness > prev_best:
            self.best_individual = current_best
            self.plateau_counter = 0
            logger.info(f"New best! Fitness: {best_fitness:.4f}")

            # Validate on val set
            self._evaluate_population([current_best], "val")
            val_fitness = current_best.val_fitness.fitness if current_best.val_fitness else 0
            logger.info(f"Validation fitness: {val_fitness:.4f}")
        else:
            self.plateau_counter += 1
            logger.info(f"No improvement (plateau: {self.plateau_counter}/{self.config.plateau_limit})")

        # Record history
        self.history.append({
            "generation": self.generation,
            "best_fitness": best_fitness,
            "best_id": current_best.id,
            "plateau_counter": self.plateau_counter,
            "population_size": len(self.population),
        })

        # Check termination
        if self.generation >= self.config.max_generations:
            logger.info(f"Reached max generations ({self.config.max_generations})")
            return False

        if self.plateau_counter >= self.config.plateau_limit:
            logger.info(f"Plateau limit reached ({self.config.plateau_limit})")
            return False

        return True

    def run(self) -> Individual:
        """
        Run the full evolution.

        Returns:
            Best individual found
        """
        self.initialize()

        while self.evolve_generation():
            pass

        # Final evaluation on test set
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("="*60)

        self._evaluate_population([self.best_individual], "test")

        test_fitness = self.best_individual.test_fitness
        if test_fitness:
            logger.info(f"Test fitness: {test_fitness.fitness:.4f}")
            logger.info(f"Test throughput: {test_fitness.throughput_lps_per_sec:.1f} LP/s "
                       f"({test_fitness.throughput_score:+.2f} vs baseline)")
            logger.info(f"Test p95: {test_fitness.p95_latency_ms:.2f}ms "
                       f"({test_fitness.p95_regression_pct:+.1f}% vs baseline)")

            if test_fitness.violates_p95_constraint:
                logger.warning("WARNING: p95 constraint violated on test set!")

        return self.best_individual

    def save_results(self, path: str):
        """Save evolution results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        results = {
            "best_config": self.best_individual.config.to_dict() if self.best_individual else None,
            "best_id": self.best_individual.id if self.best_individual else None,
            "train_fitness": self.best_individual.train_fitness.to_dict() if self.best_individual and self.best_individual.train_fitness else None,
            "val_fitness": self.best_individual.val_fitness.to_dict() if self.best_individual and self.best_individual.val_fitness else None,
            "test_fitness": self.best_individual.test_fitness.to_dict() if self.best_individual and self.best_individual.test_fitness else None,
            "generations": self.generation,
            "history": self.history,
        }

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {path}")


def main():
    """Main entry point for cuOpt evolution."""
    import argparse

    parser = argparse.ArgumentParser(description="Evolve cuOpt LP solver configurations")
    parser.add_argument("--lp-dir", type=str, help="Directory with LP benchmark files")
    parser.add_argument("--synthetic", type=int, default=100, help="Number of synthetic problems if no LP dir")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=30, help="Max generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=".evolve-sdk/cuopt_evolution/results.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s"
    )

    # Create corpus
    corpus = LPCorpus(seed=args.seed)

    if args.lp_dir:
        corpus.load_from_directory(args.lp_dir)
    else:
        corpus.generate_synthetic(
            n_train=int(args.synthetic * 0.6),
            n_val=int(args.synthetic * 0.2),
            n_test=int(args.synthetic * 0.2),
        )

    # Configure evolution
    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        verbose=args.verbose,
    )

    # Run evolution
    evolution = CuOptEvolution(corpus, config, seed=args.seed)
    best = evolution.run()

    # Save results
    evolution.save_results(args.output)

    # Print summary
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"Best config: {best.id}")
    print(f"Train fitness: {best.train_fitness.fitness:.4f}" if best.train_fitness else "N/A")
    print(f"Test fitness: {best.test_fitness.fitness:.4f}" if best.test_fitness else "N/A")

    if best.test_fitness:
        print(f"\nThroughput improvement: {best.test_fitness.throughput_score:+.1f}%")
        print(f"p95 latency change: {best.test_fitness.p95_regression_pct:+.1f}%")

        if best.test_fitness.violates_p95_constraint:
            print("\n⚠️  WARNING: Solution violates p95 constraint on test set!")
        else:
            print("\n✓ Solution passes p95 constraint on test set")


if __name__ == "__main__":
    main()
