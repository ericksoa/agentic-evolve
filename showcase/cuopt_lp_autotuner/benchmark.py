"""
benchmark.py - Evaluation harness with multi-objective fitness.

Evaluates policies across LP corpora and computes:
- Throughput (LPs/sec)
- Latency statistics (mean, p50, p95)
- Success rate
- Correctness checks

Fitness function supports:
- Primary objective: maximize throughput
- Secondary objective: minimize p95 latency
- Hard constraint: no >2% p95 regression vs baseline
- Penalties for failures and numerical errors
"""

import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .transforms import LPInstance
from .policy import Policy, load_policy, BASELINE_POLICY
from .cuopt_runner import solve_with_policy, SolveStatus, BatchSolveResult
from .feature_extract import extract_features_from_lp, quick_features
from .lp_generator import create_corpus, LPGenerator, GeneratorConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics from evaluating a policy on a corpus."""
    # Throughput
    throughput_lps_per_sec: float = 0.0

    # Latency stats (in ms)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0

    # Success metrics
    n_total: int = 0
    n_optimal: int = 0
    n_infeasible: int = 0
    n_timeout: int = 0
    n_error: int = 0
    success_rate: float = 0.0
    fail_rate: float = 0.0

    # Correctness (vs baseline)
    n_correct: int = 0
    n_incorrect: int = 0
    correctness_rate: float = 0.0
    max_obj_deviation: float = 0.0
    max_residual: float = 0.0

    # Timing
    total_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FitnessResult:
    """Multi-objective fitness result."""
    # Primary fitness (higher is better)
    primary_fitness: float = 0.0

    # Component scores
    throughput_score: float = 0.0
    latency_score: float = 0.0
    correctness_penalty: float = 0.0
    failure_penalty: float = 0.0

    # Constraint violations
    p95_regression_pct: float = 0.0  # % regression vs baseline
    violates_p95_constraint: bool = False

    # Raw metrics
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    baseline_metrics: Optional[BenchmarkMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'primary_fitness': float(self.primary_fitness),
            'throughput_score': float(self.throughput_score),
            'latency_score': float(self.latency_score),
            'correctness_penalty': float(self.correctness_penalty),
            'failure_penalty': float(self.failure_penalty),
            'p95_regression_pct': float(self.p95_regression_pct),
            'violates_p95_constraint': bool(self.violates_p95_constraint),
            'metrics': self.metrics.to_dict(),
        }
        if self.baseline_metrics:
            d['baseline_metrics'] = self.baseline_metrics.to_dict()
        return d


class Benchmark:
    """
    Benchmark harness for evaluating LP solver policies.

    Usage:
        benchmark = Benchmark(corpus)
        result = benchmark.evaluate(policy)
        fitness = benchmark.compute_fitness(policy)
    """

    def __init__(
        self,
        corpus: Dict[str, List[LPInstance]],
        baseline_policy: Optional[Policy] = None,
        timeout_ms: float = 60000,
        obj_tolerance: float = 1e-4,
        residual_tolerance: float = 1e-6,
        p95_regression_limit: float = 0.02,  # 2%
        verbose: bool = False,
    ):
        """
        Initialize benchmark.

        Args:
            corpus: Dict with 'train', 'valid', 'test' LP lists
            baseline_policy: Baseline for comparison (default: BASELINE_POLICY)
            timeout_ms: Timeout per corpus evaluation
            obj_tolerance: Relative tolerance for objective correctness
            residual_tolerance: Tolerance for primal feasibility
            p95_regression_limit: Max allowed p95 regression (fraction)
            verbose: Enable verbose output
        """
        self.corpus = corpus
        self.baseline_policy = baseline_policy or Policy.baseline()
        self.timeout_ms = timeout_ms
        self.obj_tolerance = obj_tolerance
        self.residual_tolerance = residual_tolerance
        self.p95_regression_limit = p95_regression_limit
        self.verbose = verbose

        # Pre-compute features for all LPs
        self.features = {}
        for split, lps in corpus.items():
            self.features[split] = [quick_features(lp) for lp in lps]

        # Cache baseline results
        self._baseline_cache = {}

    def evaluate(
        self,
        policy: Policy,
        split: str = 'train',
        compute_correctness: bool = True,
    ) -> BenchmarkMetrics:
        """
        Evaluate a policy on a corpus split.

        Args:
            policy: Policy to evaluate
            split: 'train', 'valid', or 'test'
            compute_correctness: Whether to check correctness vs baseline

        Returns:
            BenchmarkMetrics
        """
        lps = self.corpus[split]
        features_list = self.features[split]

        if not lps:
            return BenchmarkMetrics()

        start_time = time.perf_counter()

        # Solve all LPs
        batch_result = solve_with_policy(
            lps=lps,
            policy=policy,
            features_list=features_list,
            timeout_ms=self.timeout_ms,
            verbose=self.verbose,
        )

        total_time = time.perf_counter() - start_time

        # Compute metrics
        metrics = self._compute_metrics(batch_result, total_time)

        # Correctness check
        if compute_correctness:
            baseline_result = self._get_baseline_result(split)
            self._compute_correctness(metrics, batch_result, baseline_result)

        return metrics

    def _compute_metrics(
        self,
        result: BatchSolveResult,
        total_time: float,
    ) -> BenchmarkMetrics:
        """Compute metrics from solve results."""
        n_total = len(result.results)

        if n_total == 0:
            return BenchmarkMetrics()

        # Count status types
        n_optimal = sum(1 for r in result.results if r.status == SolveStatus.OPTIMAL)
        n_infeasible = sum(1 for r in result.results if r.status == SolveStatus.INFEASIBLE)
        n_timeout = sum(1 for r in result.results if r.status == SolveStatus.TIMEOUT)
        n_error = sum(1 for r in result.results if r.status in [
            SolveStatus.NUMERICAL_ERROR, SolveStatus.UNKNOWN
        ])

        # Latency stats (only for solved instances)
        latencies = [r.solve_time_ms for r in result.results if r.solve_time_ms > 0]

        if latencies:
            latency_mean = np.mean(latencies)
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)
            latency_max = np.max(latencies)
        else:
            latency_mean = latency_p50 = latency_p95 = latency_p99 = latency_max = 0

        return BenchmarkMetrics(
            throughput_lps_per_sec=n_total / total_time if total_time > 0 else 0,
            latency_mean_ms=latency_mean,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            latency_max_ms=latency_max,
            n_total=n_total,
            n_optimal=n_optimal,
            n_infeasible=n_infeasible,
            n_timeout=n_timeout,
            n_error=n_error,
            success_rate=n_optimal / n_total if n_total > 0 else 0,
            fail_rate=(n_timeout + n_error) / n_total if n_total > 0 else 0,
            total_time_sec=total_time,
        )

    def _compute_correctness(
        self,
        metrics: BenchmarkMetrics,
        result: BatchSolveResult,
        baseline_result: BatchSolveResult,
    ):
        """Compare results to baseline for correctness."""
        n_correct = 0
        n_incorrect = 0
        max_obj_dev = 0.0
        max_residual = 0.0

        for r, br in zip(result.results, baseline_result.results):
            if r.status != SolveStatus.OPTIMAL or br.status != SolveStatus.OPTIMAL:
                continue

            # Check objective agreement
            if br.objective is not None and r.objective is not None:
                if br.objective != 0:
                    rel_diff = abs(r.objective - br.objective) / abs(br.objective)
                else:
                    rel_diff = abs(r.objective - br.objective)

                max_obj_dev = max(max_obj_dev, rel_diff)

                if rel_diff > self.obj_tolerance:
                    n_incorrect += 1
                    continue

            # Check primal feasibility
            if r.primal_residual > self.residual_tolerance:
                n_incorrect += 1
                max_residual = max(max_residual, r.primal_residual)
                continue

            n_correct += 1

        total_comparable = n_correct + n_incorrect
        metrics.n_correct = n_correct
        metrics.n_incorrect = n_incorrect
        metrics.correctness_rate = n_correct / total_comparable if total_comparable > 0 else 1.0
        metrics.max_obj_deviation = max_obj_dev
        metrics.max_residual = max_residual

    def _get_baseline_result(self, split: str) -> BatchSolveResult:
        """Get cached baseline results for a split."""
        if split not in self._baseline_cache:
            lps = self.corpus[split]
            features_list = self.features[split]

            self._baseline_cache[split] = solve_with_policy(
                lps=lps,
                policy=self.baseline_policy,
                features_list=features_list,
                timeout_ms=self.timeout_ms,
                verbose=self.verbose,
            )

        return self._baseline_cache[split]

    def compute_fitness(
        self,
        policy: Policy,
        split: str = 'train',
    ) -> FitnessResult:
        """
        Compute multi-objective fitness for a policy.

        Fitness formula:
            fitness = throughput_score + latency_score - penalties

        Where:
            throughput_score = (throughput / baseline_throughput - 1) * 100
            latency_score = (1 - p95 / baseline_p95) * 50
            penalties = failure_penalty + correctness_penalty

        Args:
            policy: Policy to evaluate
            split: Corpus split to use

        Returns:
            FitnessResult with detailed breakdown
        """
        # Check if this is the baseline policy (avoid self-comparison timing issues)
        is_baseline = policy.name == 'baseline'

        # Evaluate policy
        metrics = self.evaluate(policy, split, compute_correctness=not is_baseline)

        # Get baseline metrics
        baseline_metrics = self._get_baseline_metrics(split)

        # If evaluating baseline, use same metrics to avoid timing noise
        if is_baseline:
            metrics = baseline_metrics

        # Compute component scores
        throughput_score = self._compute_throughput_score(metrics, baseline_metrics)
        latency_score = self._compute_latency_score(metrics, baseline_metrics)
        failure_penalty = self._compute_failure_penalty(metrics, baseline_metrics)
        correctness_penalty = self._compute_correctness_penalty(metrics)

        # Check p95 constraint (skip for baseline - always passes)
        if is_baseline:
            p95_regression = 0.0
            violates_constraint = False
        else:
            p95_regression = self._compute_p95_regression(metrics, baseline_metrics)
            violates_constraint = p95_regression > self.p95_regression_limit

        # Final fitness
        primary_fitness = throughput_score + latency_score - failure_penalty - correctness_penalty

        # If constraint violated, fitness is very negative
        if violates_constraint:
            primary_fitness = -1000 - p95_regression * 1000

        return FitnessResult(
            primary_fitness=primary_fitness,
            throughput_score=throughput_score,
            latency_score=latency_score,
            correctness_penalty=correctness_penalty,
            failure_penalty=failure_penalty,
            p95_regression_pct=p95_regression * 100,
            violates_p95_constraint=violates_constraint,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
        )

    def _get_baseline_metrics(self, split: str) -> BenchmarkMetrics:
        """Get baseline metrics for a split."""
        # Force baseline evaluation
        self._get_baseline_result(split)

        lps = self.corpus[split]
        start = time.perf_counter()
        result = self._baseline_cache[split]
        elapsed = time.perf_counter() - start

        # Compute metrics (simplified - use cached result time)
        return self._compute_metrics(result, result.total_time_ms / 1000)

    def _compute_throughput_score(
        self,
        metrics: BenchmarkMetrics,
        baseline: BenchmarkMetrics,
    ) -> float:
        """Compute throughput improvement score."""
        if baseline.throughput_lps_per_sec <= 0:
            return 0.0

        improvement = (metrics.throughput_lps_per_sec / baseline.throughput_lps_per_sec) - 1
        return improvement * 100  # Scale to percentage points

    def _compute_latency_score(
        self,
        metrics: BenchmarkMetrics,
        baseline: BenchmarkMetrics,
    ) -> float:
        """Compute latency improvement score."""
        if baseline.latency_p95_ms <= 0:
            return 0.0

        # Lower is better, so improvement is (1 - ratio)
        improvement = 1 - (metrics.latency_p95_ms / baseline.latency_p95_ms)
        return improvement * 50  # Scale factor

    def _compute_failure_penalty(
        self,
        metrics: BenchmarkMetrics,
        baseline: BenchmarkMetrics,
    ) -> float:
        """Compute penalty for failures exceeding baseline."""
        # Penalty if fail rate is higher than baseline
        excess_failures = metrics.fail_rate - baseline.fail_rate
        if excess_failures > 0:
            return excess_failures * 200  # Heavy penalty
        return 0.0

    def _compute_correctness_penalty(self, metrics: BenchmarkMetrics) -> float:
        """Compute penalty for incorrect solutions."""
        if metrics.n_correct + metrics.n_incorrect == 0:
            return 0.0

        incorrect_rate = metrics.n_incorrect / (metrics.n_correct + metrics.n_incorrect)
        return incorrect_rate * 100

    def _compute_p95_regression(
        self,
        metrics: BenchmarkMetrics,
        baseline: BenchmarkMetrics,
    ) -> float:
        """Compute p95 regression as fraction."""
        if baseline.latency_p95_ms <= 0:
            return 0.0

        regression = (metrics.latency_p95_ms - baseline.latency_p95_ms) / baseline.latency_p95_ms
        return max(0, regression)  # Only count positive regression


def evaluate_policy(
    policy_path: str,
    corpus_config: Optional[Dict] = None,
    split: str = 'test',
    seed: int = 42,
    verbose: bool = False,
) -> FitnessResult:
    """
    Convenience function to evaluate a policy.

    Args:
        policy_path: Path to policy JSON or 'baseline'
        corpus_config: Generator config (uses defaults if None)
        split: Which split to evaluate on
        seed: Random seed for corpus generation
        verbose: Enable verbose output

    Returns:
        FitnessResult
    """
    # Load policy
    policy = load_policy(policy_path)

    # Create corpus
    gen_config = GeneratorConfig(**(corpus_config or {}))
    corpus = create_corpus(seed=seed, config=gen_config)

    # Create benchmark and evaluate
    benchmark = Benchmark(corpus, verbose=verbose)
    return benchmark.compute_fitness(policy, split)


def run_benchmark(
    policy_path: str = 'baseline',
    n_train: int = 50,
    n_test: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a full benchmark and return results.

    This is the main entry point for CLI usage.
    """
    print(f"Loading policy: {policy_path}")
    policy = load_policy(policy_path)

    print(f"Generating corpus (seed={seed})...")
    corpus = create_corpus(n_train=n_train, n_valid=10, n_test=n_test, seed=seed)

    print(f"  Train: {len(corpus['train'])} LPs")
    print(f"  Valid: {len(corpus['valid'])} LPs")
    print(f"  Test: {len(corpus['test'])} LPs")

    benchmark = Benchmark(corpus, verbose=verbose)

    results = {}
    for split in ['train', 'test']:
        print(f"\nEvaluating on {split}...")
        fitness = benchmark.compute_fitness(policy, split)

        results[split] = fitness.to_dict()

        print(f"  Throughput: {fitness.metrics.throughput_lps_per_sec:.2f} LPs/sec")
        print(f"  Latency p95: {fitness.metrics.latency_p95_ms:.2f} ms")
        print(f"  Success rate: {fitness.metrics.success_rate:.1%}")
        print(f"  Fitness: {fitness.primary_fitness:.2f}")

        if fitness.violates_p95_constraint:
            print(f"  WARNING: p95 regression {fitness.p95_regression_pct:.1f}% exceeds limit!")

    return results


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark LP solver policies')
    parser.add_argument('--policy', type=str, default='baseline',
                        help='Policy path or "baseline"')
    parser.add_argument('--n-train', type=int, default=50,
                        help='Number of training instances')
    parser.add_argument('--n-test', type=int, default=20,
                        help='Number of test instances')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    results = run_benchmark(
        policy_path=args.policy,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
