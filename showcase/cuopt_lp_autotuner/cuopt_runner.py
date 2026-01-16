"""
cuopt_runner.py - cuOpt LP solver adapter with mock fallback.

This module provides a unified interface for solving LPs:
1. If cuOpt is available, use it with the specified parameters
2. Otherwise, fall back to scipy.optimize.linprog (for testing/development)

The interface is designed so that swapping to real cuOpt requires minimal changes.

Key features:
- Batch solving support (multiple LPs in one call)
- Warm-start support (if available)
- Configurable timeouts
- Detailed solve statistics
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging

from .transforms import LPInstance, TransformPipeline, get_safe_transform_specs

logger = logging.getLogger(__name__)


class SolveStatus(Enum):
    """LP solve status codes."""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    NUMERICAL_ERROR = "numerical_error"
    UNKNOWN = "unknown"


@dataclass
class SolveResult:
    """Result of solving a single LP."""
    status: SolveStatus
    objective: Optional[float] = None
    solution: Optional[np.ndarray] = None
    solve_time_ms: float = 0.0
    iterations: int = 0
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    error_message: str = ""


@dataclass
class BatchSolveResult:
    """Result of solving a batch of LPs."""
    results: List[SolveResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    throughput_lps_per_sec: float = 0.0
    success_rate: float = 0.0


# Check if cuOpt is available
CUOPT_AVAILABLE = False
try:
    # TODO: Import actual cuOpt when available
    # from cuopt import LinearProgramSolver
    # CUOPT_AVAILABLE = True
    pass
except ImportError:
    pass


def check_cuopt_available() -> bool:
    """Check if cuOpt GPU solver is available."""
    return CUOPT_AVAILABLE


class LPSolver:
    """
    LP Solver wrapper with cuOpt and scipy fallback.

    Usage:
        solver = LPSolver(use_gpu=True)
        result = solver.solve(lp, params)
        batch_result = solver.solve_batch(lps, params)
    """

    def __init__(self, use_gpu: bool = True, verbose: bool = False):
        """
        Initialize solver.

        Args:
            use_gpu: Whether to try using cuOpt (falls back to scipy if unavailable)
            verbose: Enable verbose logging
        """
        self.use_gpu = use_gpu and CUOPT_AVAILABLE
        self.verbose = verbose

        if use_gpu and not CUOPT_AVAILABLE:
            logger.info("cuOpt not available, using scipy fallback")

    def solve(
        self,
        lp: LPInstance,
        params: Optional[Dict[str, Any]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        timeout_ms: float = 30000,
    ) -> Tuple[SolveResult, Optional[TransformPipeline]]:
        """
        Solve a single LP instance.

        Args:
            lp: LP instance to solve
            params: Solver parameters (tolerance, max_iterations, etc.)
            transform_config: Transform configuration (enable_row_scale, etc.)
            timeout_ms: Timeout in milliseconds

        Returns:
            (SolveResult, TransformPipeline used)
        """
        params = params or {}
        transform_config = transform_config or {}

        # Apply transforms
        pipeline = TransformPipeline()
        transform_specs = get_safe_transform_specs(**transform_config)

        if transform_specs:
            lp_transformed = pipeline.apply(lp, transform_specs)
        else:
            lp_transformed = lp

        # Solve
        start_time = time.perf_counter()

        if self.use_gpu:
            result = self._solve_cuopt(lp_transformed, params, timeout_ms)
        else:
            result = self._solve_scipy(lp_transformed, params, timeout_ms)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.solve_time_ms = elapsed_ms

        # Map solution back to original space
        if result.status == SolveStatus.OPTIMAL and result.solution is not None:
            result.solution = pipeline.inverse_solution(result.solution)
            result.objective = pipeline.inverse_objective(result.objective)

            # Verify solution feasibility
            residual = self._compute_primal_residual(lp, result.solution)
            result.primal_residual = residual

        return result, pipeline

    def solve_batch(
        self,
        lps: List[LPInstance],
        params: Optional[Dict[str, Any]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        timeout_ms: float = 60000,
    ) -> BatchSolveResult:
        """
        Solve a batch of LP instances.

        For GPU efficiency, we batch similar-sized problems together.

        Args:
            lps: List of LP instances
            params: Solver parameters
            transform_config: Transform configuration
            timeout_ms: Total timeout for batch

        Returns:
            BatchSolveResult with individual results and aggregate stats
        """
        if not lps:
            return BatchSolveResult()

        start_time = time.perf_counter()
        results = []

        # For now, solve sequentially (TODO: true batching with cuOpt)
        per_lp_timeout = timeout_ms / len(lps)

        for lp in lps:
            result, _ = self.solve(lp, params, transform_config, per_lp_timeout)
            results.append(result)

            # Check total timeout
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > timeout_ms:
                # Fill remaining with timeout status
                while len(results) < len(lps):
                    results.append(SolveResult(
                        status=SolveStatus.TIMEOUT,
                        error_message="Batch timeout exceeded",
                    ))
                break

        total_time_ms = (time.perf_counter() - start_time) * 1000
        n_success = sum(1 for r in results if r.status == SolveStatus.OPTIMAL)

        return BatchSolveResult(
            results=results,
            total_time_ms=total_time_ms,
            throughput_lps_per_sec=len(lps) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
            success_rate=n_success / len(lps) if lps else 0,
        )

    def _solve_cuopt(
        self,
        lp: LPInstance,
        params: Dict[str, Any],
        timeout_ms: float,
    ) -> SolveResult:
        """
        Solve using cuOpt GPU solver.

        TODO: Implement actual cuOpt integration when API is available.

        Expected cuOpt API (placeholder):
            solver = cuopt.LinearProgramSolver()
            solver.set_problem(c, A, b, lb, ub, sense)
            solver.set_params(tolerance=..., max_iter=...)
            result = solver.solve()
        """
        # Placeholder - fall back to scipy for now
        logger.warning("cuOpt not implemented, falling back to scipy")
        return self._solve_scipy(lp, params, timeout_ms)

    def _solve_scipy(
        self,
        lp: LPInstance,
        params: Dict[str, Any],
        timeout_ms: float,
    ) -> SolveResult:
        """
        Solve using scipy.optimize.linprog (fallback solver).

        scipy.linprog only supports <= constraints and bounds, so we need
        to convert the LP to standard form.
        """
        try:
            from scipy.optimize import linprog
            from scipy.sparse import csr_matrix
        except ImportError:
            return SolveResult(
                status=SolveStatus.UNKNOWN,
                error_message="scipy not available",
            )

        try:
            # Convert to scipy format
            # scipy expects: min c'x s.t. A_ub @ x <= b_ub, A_eq @ x = b_eq, bounds

            n_vars = lp.n_vars
            n_cons = lp.n_cons

            # Separate constraints by sense
            eq_mask = lp.sense == 0
            leq_mask = lp.sense == -1
            geq_mask = lp.sense == 1

            A_full = lp.to_csr()

            # Equality constraints
            A_eq = A_full[eq_mask] if np.any(eq_mask) else None
            b_eq = lp.b[eq_mask] if np.any(eq_mask) else None

            # Inequality constraints (convert >= to <= by negation)
            A_ub_parts = []
            b_ub_parts = []

            if np.any(leq_mask):
                A_ub_parts.append(A_full[leq_mask])
                b_ub_parts.append(lp.b[leq_mask])

            if np.any(geq_mask):
                A_ub_parts.append(-A_full[geq_mask])
                b_ub_parts.append(-lp.b[geq_mask])

            if A_ub_parts:
                from scipy.sparse import vstack
                A_ub = vstack(A_ub_parts)
                b_ub = np.concatenate(b_ub_parts)
            else:
                A_ub = None
                b_ub = None

            # Bounds
            bounds = list(zip(lp.lb, lp.ub))

            # Solver options (HiGHS uses different parameter names)
            options = {
                'maxiter': params.get('max_iterations', 10000),
                'presolve': params.get('presolve', True),
                'primal_feasibility_tolerance': params.get('tolerance', 1e-6),
                'dual_feasibility_tolerance': params.get('tolerance', 1e-6),
            }

            # Solve
            result = linprog(
                c=lp.c,
                A_ub=A_ub.toarray() if A_ub is not None else None,
                b_ub=b_ub,
                A_eq=A_eq.toarray() if A_eq is not None else None,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options=options,
            )

            # Convert result
            if result.success:
                return SolveResult(
                    status=SolveStatus.OPTIMAL,
                    objective=float(result.fun),
                    solution=result.x,
                    iterations=result.nit if hasattr(result, 'nit') else 0,
                )
            elif 'infeasible' in result.message.lower():
                return SolveResult(
                    status=SolveStatus.INFEASIBLE,
                    error_message=result.message,
                )
            elif 'unbounded' in result.message.lower():
                return SolveResult(
                    status=SolveStatus.UNBOUNDED,
                    error_message=result.message,
                )
            else:
                return SolveResult(
                    status=SolveStatus.UNKNOWN,
                    error_message=result.message,
                )

        except Exception as e:
            return SolveResult(
                status=SolveStatus.NUMERICAL_ERROR,
                error_message=str(e),
            )

    def _compute_primal_residual(self, lp: LPInstance, x: np.ndarray) -> float:
        """
        Compute primal feasibility residual.

        Returns max violation of Ax {<=,=,>=} b constraints.
        """
        try:
            A = lp.to_csr()
            Ax = A @ x

            residuals = []
            for i in range(lp.n_cons):
                ax_i = Ax[i]
                b_i = lp.b[i]
                sense_i = lp.sense[i]

                if sense_i == -1:  # <=
                    residuals.append(max(0, ax_i - b_i))
                elif sense_i == 1:  # >=
                    residuals.append(max(0, b_i - ax_i))
                else:  # ==
                    residuals.append(abs(ax_i - b_i))

            # Also check bound violations
            lb_viol = np.maximum(0, lp.lb - x)
            ub_viol = np.maximum(0, x - lp.ub)

            # Handle infinities
            lb_viol = np.where(np.isfinite(lb_viol), lb_viol, 0)
            ub_viol = np.where(np.isfinite(ub_viol), ub_viol, 0)

            max_constraint_viol = max(residuals) if residuals else 0
            max_bound_viol = max(np.max(lb_viol), np.max(ub_viol))

            return max(max_constraint_viol, max_bound_viol)

        except Exception:
            return float('inf')


class BatchingStrategy:
    """
    Groups LPs into batches for efficient GPU solving.

    Strategies:
    - size_homogeneous: group by similar n_vars + n_cons
    - density_homogeneous: group by similar density
    - mixed: group by both size and density buckets
    """

    def __init__(
        self,
        size_thresholds: List[float] = [100, 1000, 10000],
        density_thresholds: List[float] = [0.01, 0.1],
        max_batch_size: int = 32,
        strategy: str = 'size_homogeneous',
    ):
        self.size_thresholds = sorted(size_thresholds)
        self.density_thresholds = sorted(density_thresholds)
        self.max_batch_size = max_batch_size
        self.strategy = strategy

    def create_batches(self, lps: List[LPInstance]) -> List[List[int]]:
        """
        Group LP indices into batches.

        Args:
            lps: List of LP instances

        Returns:
            List of batches, each batch is a list of indices
        """
        if self.strategy == 'none' or not lps:
            # No batching - each LP is its own batch
            return [[i] for i in range(len(lps))]

        # Compute bucket for each LP
        buckets = {}
        for i, lp in enumerate(lps):
            bucket_key = self._get_bucket_key(lp)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(i)

        # Create batches within each bucket
        batches = []
        for bucket_indices in buckets.values():
            for j in range(0, len(bucket_indices), self.max_batch_size):
                batch = bucket_indices[j:j + self.max_batch_size]
                batches.append(batch)

        return batches

    def _get_bucket_key(self, lp: LPInstance) -> Tuple:
        """Get bucket key for an LP based on strategy."""
        size = lp.n_vars + lp.n_cons
        density = lp.nnz / (lp.n_vars * lp.n_cons) if lp.n_vars > 0 and lp.n_cons > 0 else 0

        size_bucket = self._get_bucket_index(size, self.size_thresholds)
        density_bucket = self._get_bucket_index(density, self.density_thresholds)

        if self.strategy == 'size_homogeneous':
            return (size_bucket,)
        elif self.strategy == 'density_homogeneous':
            return (density_bucket,)
        else:  # mixed
            return (size_bucket, density_bucket)

    def _get_bucket_index(self, value: float, thresholds: List[float]) -> int:
        """Get bucket index for a value given thresholds."""
        for i, thresh in enumerate(thresholds):
            if value < thresh:
                return i
        return len(thresholds)


def solve_with_policy(
    lps: List[LPInstance],
    policy: 'Policy',
    features_list: Optional[List[Dict]] = None,
    timeout_ms: float = 60000,
    verbose: bool = False,
) -> BatchSolveResult:
    """
    Solve LPs using a policy to select transforms and parameters.

    This is the main entry point for benchmarking a policy.

    Args:
        lps: LP instances to solve
        policy: Policy specifying transforms, params, and batching
        features_list: Pre-computed features for each LP (optional)
        timeout_ms: Total timeout
        verbose: Enable verbose output

    Returns:
        BatchSolveResult with all results and aggregate stats
    """
    from .policy import Policy

    if not lps:
        return BatchSolveResult()

    # Get batching config (use features of first LP if available)
    sample_features = features_list[0] if features_list else None
    batching_config = policy.get_batching_config(sample_features)

    # Create batches
    batcher = BatchingStrategy(
        size_thresholds=batching_config.get('size_thresholds', [100, 1000, 10000]),
        density_thresholds=batching_config.get('density_thresholds', [0.01, 0.1]),
        max_batch_size=batching_config.get('max_batch_size', 1),
        strategy=batching_config.get('strategy', 'none') if batching_config.get('enabled', False) else 'none',
    )

    batches = batcher.create_batches(lps)

    # Solve each batch
    solver = LPSolver(use_gpu=True, verbose=verbose)
    all_results = [None] * len(lps)

    start_time = time.perf_counter()

    for batch_indices in batches:
        # Get LP-specific configs (use first LP's features for batch)
        batch_features = features_list[batch_indices[0]] if features_list else None
        transform_config = policy.get_transform_config(batch_features)
        solver_params = policy.get_solver_params(batch_features)

        batch_lps = [lps[i] for i in batch_indices]
        remaining_time = timeout_ms - (time.perf_counter() - start_time) * 1000

        if remaining_time <= 0:
            for i in batch_indices:
                all_results[i] = SolveResult(status=SolveStatus.TIMEOUT)
            continue

        batch_result = solver.solve_batch(
            batch_lps,
            params=solver_params,
            transform_config=transform_config,
            timeout_ms=remaining_time,
        )

        for j, i in enumerate(batch_indices):
            all_results[i] = batch_result.results[j]

    total_time_ms = (time.perf_counter() - start_time) * 1000
    n_success = sum(1 for r in all_results if r and r.status == SolveStatus.OPTIMAL)

    return BatchSolveResult(
        results=all_results,
        total_time_ms=total_time_ms,
        throughput_lps_per_sec=len(lps) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        success_rate=n_success / len(lps) if lps else 0,
    )
