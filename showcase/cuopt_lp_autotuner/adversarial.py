"""
adversarial.py - Adversarial instance generator.

Generates hard-but-plausible LP instances that stress the current best policy.
Uses a minimax approach: search generator parameter space to find instances
that maximize the gap between policy performance and baseline.

This helps prevent overfitting by continuously challenging the evolved policy
with edge cases and difficult scenarios.
"""

import numpy as np
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .transforms import LPInstance
from .policy import Policy
from .lp_generator import LPGenerator, GeneratorConfig, LPType
from .cuopt_runner import solve_with_policy, SolveStatus
from .feature_extract import quick_features

logger = logging.getLogger(__name__)


@dataclass
class AdversarialConfig:
    """Configuration for adversarial instance generation."""

    # Generator parameter ranges to search
    size_range: Tuple[int, int] = (50, 5000)
    density_range: Tuple[float, float] = (0.001, 0.5)
    coef_range_log: Tuple[float, float] = (0, 8)  # 10^0 to 10^8
    rhs_scale_range: Tuple[float, float] = (1, 1e6)

    # Search parameters
    n_candidates: int = 20
    n_mutations: int = 5
    mutation_strength: float = 0.3

    # Scoring weights
    weight_solve_time: float = 1.0
    weight_failure: float = 10.0
    weight_regression: float = 5.0


@dataclass
class GeneratorGenome:
    """Genome representing generator parameters."""

    # Size parameters
    n_vars: int = 100
    n_cons: int = 50
    density: float = 0.1

    # Coefficient parameters
    coef_min: float = 0.1
    coef_max: float = 100.0

    # RHS parameters
    rhs_scale: float = 100.0

    # Structure parameters
    frac_leq: float = 0.6
    frac_eq: float = 0.2
    frac_bounded: float = 0.7

    # Problem type
    problem_type: str = "generic"

    def to_config(self) -> GeneratorConfig:
        """Convert to GeneratorConfig."""
        return GeneratorConfig(
            min_vars=max(5, self.n_vars - 10),
            max_vars=self.n_vars + 10,
            min_cons=max(3, self.n_cons - 5),
            max_cons=self.n_cons + 5,
            min_density=max(0.001, self.density * 0.8),
            max_density=min(0.9, self.density * 1.2),
            coef_min=self.coef_min,
            coef_max=self.coef_max,
            rhs_scale=self.rhs_scale,
            frac_leq=self.frac_leq,
            frac_eq=self.frac_eq,
            frac_bounded=self.frac_bounded,
            problem_types=[self.problem_type],
        )

    def mutate(self, strength: float, rng: random.Random) -> 'GeneratorGenome':
        """Create a mutated copy."""
        new = GeneratorGenome(
            n_vars=self.n_vars,
            n_cons=self.n_cons,
            density=self.density,
            coef_min=self.coef_min,
            coef_max=self.coef_max,
            rhs_scale=self.rhs_scale,
            frac_leq=self.frac_leq,
            frac_eq=self.frac_eq,
            frac_bounded=self.frac_bounded,
            problem_type=self.problem_type,
        )

        # Mutate size
        if rng.random() < strength:
            new.n_vars = max(10, int(self.n_vars * (1 + rng.gauss(0, 0.5))))
        if rng.random() < strength:
            new.n_cons = max(5, int(self.n_cons * (1 + rng.gauss(0, 0.5))))

        # Mutate density
        if rng.random() < strength:
            new.density = max(0.001, min(0.9, self.density * (1 + rng.gauss(0, 0.3))))

        # Mutate coefficients (in log space)
        if rng.random() < strength:
            log_min = np.log10(self.coef_min)
            log_min += rng.gauss(0, 1)
            new.coef_min = max(1e-10, 10 ** log_min)
        if rng.random() < strength:
            log_max = np.log10(self.coef_max)
            log_max += rng.gauss(0, 1)
            new.coef_max = min(1e15, 10 ** log_max)

        # Ensure min < max
        if new.coef_min >= new.coef_max:
            new.coef_min, new.coef_max = new.coef_max / 10, new.coef_max

        # Mutate RHS scale
        if rng.random() < strength:
            log_rhs = np.log10(self.rhs_scale)
            log_rhs += rng.gauss(0, 1)
            new.rhs_scale = max(1, min(1e10, 10 ** log_rhs))

        # Mutate constraint distribution
        if rng.random() < strength:
            new.frac_leq = max(0, min(1, self.frac_leq + rng.gauss(0, 0.2)))
            new.frac_eq = max(0, min(1 - new.frac_leq, self.frac_eq + rng.gauss(0, 0.1)))

        # Mutate problem type
        if rng.random() < strength * 0.5:
            types = [t.value for t in LPType]
            new.problem_type = rng.choice(types)

        return new


class AdversarialGenerator:
    """
    Generates adversarial LP instances.

    Uses evolutionary search over generator parameters to find
    instances that are hard for the target policy but solvable
    by the baseline.
    """

    def __init__(
        self,
        target_policy: Policy,
        benchmark: 'Benchmark',
        seed: int = 42,
        config: Optional[AdversarialConfig] = None,
    ):
        """
        Initialize adversarial generator.

        Args:
            target_policy: Policy to attack
            benchmark: Benchmark instance for evaluation
            seed: Random seed
            config: Adversarial configuration
        """
        self.target_policy = target_policy
        self.benchmark = benchmark
        self.baseline_policy = Policy.baseline()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.config = config or AdversarialConfig()

    def generate(self, n_instances: int) -> List[LPInstance]:
        """
        Generate adversarial instances.

        Uses a simple evolutionary search:
        1. Generate random generator genomes
        2. Score each by how much it hurts the target policy
        3. Keep the best and mutate to find harder instances
        4. Return instances from the hardest genomes

        Args:
            n_instances: Number of instances to generate

        Returns:
            List of adversarial LP instances
        """
        logger.info(f"Generating {n_instances} adversarial instances")

        # Initialize population of generator genomes
        population = self._initialize_genomes()

        # Score initial population
        scored = [(g, self._score_genome(g)) for g in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Evolutionary improvement
        for iteration in range(3):  # Quick iterations
            # Keep top half
            survivors = scored[:len(scored) // 2]

            # Generate offspring
            offspring = []
            for genome, _ in survivors:
                for _ in range(self.config.n_mutations):
                    child = genome.mutate(self.config.mutation_strength, self.rng)
                    offspring.append(child)

            # Score offspring
            for genome in offspring:
                score = self._score_genome(genome)
                scored.append((genome, score))

            # Select top individuals
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:self.config.n_candidates]

        # Generate instances from top genomes
        instances = []
        for genome, score in scored[:n_instances]:
            if score > 0:  # Only include if actually adversarial
                lp = self._generate_from_genome(genome)
                if lp is not None:
                    instances.append(lp)

        logger.info(f"Generated {len(instances)} adversarial instances")
        return instances

    def _initialize_genomes(self) -> List[GeneratorGenome]:
        """Initialize a diverse population of generator genomes."""
        genomes = []

        # Include some targeted configurations
        targeted_configs = [
            # Large sparse problems
            {'n_vars': 2000, 'n_cons': 1000, 'density': 0.01},
            # Small dense problems
            {'n_vars': 50, 'n_cons': 30, 'density': 0.3},
            # Numerically challenging
            {'n_vars': 100, 'n_cons': 50, 'coef_min': 1e-6, 'coef_max': 1e6},
            # Large RHS values
            {'n_vars': 100, 'n_cons': 50, 'rhs_scale': 1e6},
            # Many equality constraints
            {'n_vars': 100, 'n_cons': 80, 'frac_eq': 0.8, 'frac_leq': 0.1},
        ]

        for config in targeted_configs:
            genomes.append(GeneratorGenome(**config))

        # Random configurations
        for _ in range(self.config.n_candidates - len(genomes)):
            genomes.append(self._random_genome())

        return genomes

    def _random_genome(self) -> GeneratorGenome:
        """Generate a random genome."""
        cfg = self.config

        n_vars = self.rng.randint(*cfg.size_range)
        n_cons = self.rng.randint(5, max(6, n_vars))

        log_min = self.rng.uniform(*cfg.coef_range_log[:1] * 2)
        log_max = log_min + self.rng.uniform(0, cfg.coef_range_log[1] - log_min)

        return GeneratorGenome(
            n_vars=n_vars,
            n_cons=n_cons,
            density=self.rng.uniform(*cfg.density_range),
            coef_min=10 ** log_min,
            coef_max=10 ** log_max,
            rhs_scale=10 ** self.rng.uniform(np.log10(cfg.rhs_scale_range[0]),
                                             np.log10(cfg.rhs_scale_range[1])),
            frac_leq=self.rng.uniform(0.2, 0.8),
            frac_eq=self.rng.uniform(0, 0.5),
            frac_bounded=self.rng.uniform(0.3, 1.0),
            problem_type=self.rng.choice([t.value for t in LPType]),
        )

    def _score_genome(self, genome: GeneratorGenome) -> float:
        """
        Score a genome by how adversarial it is.

        Higher score = more adversarial (harder for target, easier for baseline).
        """
        try:
            # Generate test instance
            lp = self._generate_from_genome(genome)
            if lp is None:
                return -1.0

            features = quick_features(lp)

            # Solve with target policy
            target_result = solve_with_policy(
                [lp], self.target_policy, [features], timeout_ms=5000
            )

            # Solve with baseline
            baseline_result = solve_with_policy(
                [lp], self.baseline_policy, [features], timeout_ms=5000
            )

            target_r = target_result.results[0]
            baseline_r = baseline_result.results[0]

            # Score components
            score = 0.0

            # Failure penalty: if target fails but baseline succeeds
            if target_r.status != SolveStatus.OPTIMAL and baseline_r.status == SolveStatus.OPTIMAL:
                score += self.config.weight_failure

            # Solve time regression
            if target_r.solve_time_ms > 0 and baseline_r.solve_time_ms > 0:
                time_ratio = target_r.solve_time_ms / baseline_r.solve_time_ms
                if time_ratio > 1:
                    score += self.config.weight_solve_time * np.log(time_ratio)

            # Objective regression (if both solved)
            if (target_r.status == SolveStatus.OPTIMAL and
                baseline_r.status == SolveStatus.OPTIMAL and
                target_r.objective is not None and baseline_r.objective is not None):

                if baseline_r.objective != 0:
                    obj_diff = abs(target_r.objective - baseline_r.objective) / abs(baseline_r.objective)
                    if obj_diff > 1e-4:
                        score += self.config.weight_regression * obj_diff

            # Don't reward instances that baseline also fails on
            if baseline_r.status != SolveStatus.OPTIMAL:
                score *= 0.1

            return score

        except Exception as e:
            logger.debug(f"Error scoring genome: {e}")
            return -1.0

    def _generate_from_genome(self, genome: GeneratorGenome) -> Optional[LPInstance]:
        """Generate an LP instance from a genome."""
        try:
            config = genome.to_config()
            gen = LPGenerator(seed=self.rng.randint(0, 10000), config=config)

            if genome.problem_type in [t.value for t in LPType]:
                lp_type = LPType(genome.problem_type)
            else:
                lp_type = None

            return gen.generate(lp_type)

        except Exception as e:
            logger.debug(f"Error generating LP: {e}")
            return None


def find_adversarial_instances(
    policy: Policy,
    corpus: Dict[str, List[LPInstance]],
    n_instances: int = 10,
    seed: int = 42,
) -> List[LPInstance]:
    """
    Convenience function to find adversarial instances.

    Args:
        policy: Target policy to attack
        corpus: Existing corpus (for benchmark creation)
        n_instances: Number of instances to find
        seed: Random seed

    Returns:
        List of adversarial instances
    """
    from .benchmark import Benchmark

    benchmark = Benchmark(corpus, verbose=False)
    generator = AdversarialGenerator(policy, benchmark, seed=seed)
    return generator.generate(n_instances)
