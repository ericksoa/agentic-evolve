"""
cuopt_solver.py - Real cuOpt LP solver integration.

This module provides the actual interface to NVIDIA cuOpt GPU-accelerated
LP solver. When cuOpt is not available, it falls back to scipy HiGHS.

Based on cuOpt 25.01+ API:
https://docs.nvidia.com/cuopt/user-guide/latest/lp-milp-settings.html
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .cuopt_config import CuOptConfig, SolverMethod, PDLPSolverMode
from .mps_loader import LPData

logger = logging.getLogger(__name__)


class SolveStatus(Enum):
    """LP solve status."""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ITERATION_LIMIT = "iteration_limit"
    NUMERICAL_ERROR = "numerical_error"
    UNKNOWN = "unknown"


@dataclass
class SolveResult:
    """Result from solving an LP."""
    status: SolveStatus
    objective: Optional[float] = None
    solution: Optional[np.ndarray] = None
    solve_time_ms: float = 0.0
    iterations: int = 0
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    gap: float = 0.0
    error_message: str = ""

    # cuOpt-specific metrics
    method_used: str = ""
    pdlp_mode_used: str = ""


# Check for cuOpt availability
CUOPT_AVAILABLE = False
cuopt_module = None

try:
    import cuopt
    CUOPT_AVAILABLE = True
    cuopt_module = cuopt
    logger.info(f"cuOpt version {cuopt.__version__} available")
except ImportError:
    logger.info("cuOpt not available, will use scipy HiGHS fallback")


def is_cuopt_available() -> bool:
    """Check if cuOpt is available."""
    return CUOPT_AVAILABLE


class CuOptSolver:
    """
    cuOpt LP solver with configurable parameters.

    Usage:
        solver = CuOptSolver()
        config = CuOptConfig(method=SolverMethod.PDLP)
        result = solver.solve(lp_data, config)
    """

    def __init__(self, use_fallback: bool = True, verbose: bool = False):
        """
        Initialize solver.

        Args:
            use_fallback: Use scipy HiGHS when cuOpt is unavailable
            verbose: Enable verbose output
        """
        self.use_fallback = use_fallback
        self.verbose = verbose
        self._cuopt_solver = None

        if CUOPT_AVAILABLE:
            try:
                self._cuopt_solver = cuopt_module.LinearProgram()
                logger.info("cuOpt LinearProgram initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize cuOpt: {e}")

    def solve(
        self,
        lp: LPData,
        config: Optional[CuOptConfig] = None,
        timeout_ms: float = 30000,
    ) -> SolveResult:
        """
        Solve an LP with the given configuration.

        Args:
            lp: LP data (from mps_loader)
            config: cuOpt configuration (uses defaults if None)
            timeout_ms: Timeout in milliseconds

        Returns:
            SolveResult with solution and metrics
        """
        if config is None:
            config = CuOptConfig.baseline()

        start_time = time.perf_counter()

        if CUOPT_AVAILABLE and self._cuopt_solver is not None:
            result = self._solve_cuopt(lp, config, timeout_ms)
        elif self.use_fallback:
            result = self._solve_scipy(lp, config, timeout_ms)
        else:
            result = SolveResult(
                status=SolveStatus.UNKNOWN,
                error_message="cuOpt not available and fallback disabled",
            )

        result.solve_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _solve_cuopt(
        self,
        lp: LPData,
        config: CuOptConfig,
        timeout_ms: float,
    ) -> SolveResult:
        """
        Solve using NVIDIA cuOpt.

        cuOpt API (25.01+):
            solver = cuopt.LinearProgram()
            solver.set_problem(c, A, b, sense, lb, ub)
            solver.set_params(**config.to_cuopt_params())
            result = solver.solve()
        """
        try:
            solver = cuopt_module.LinearProgram()

            # Get LP data in cuOpt format
            lp_dict = lp.to_cuopt_format()
            c = lp_dict["c"]
            A = lp_dict["A"]
            b = lp_dict["b"]
            sense = lp_dict["sense"]
            lb = lp_dict["lb"]
            ub = lp_dict["ub"]

            # Set the problem
            # cuOpt sense: -1 = <=, 0 = =, 1 = >=
            solver.set_problem(
                c=c.astype(np.float64),
                A=A.tocsr(),
                b=b.astype(np.float64),
                constraint_sense=sense.astype(np.int32),
                lb=lb.astype(np.float64),
                ub=ub.astype(np.float64),
            )

            # Set parameters from config
            params = config.to_cuopt_params()

            # Override time limit with our timeout
            params["CUOPT_TIME_LIMIT"] = timeout_ms / 1000.0

            for key, value in params.items():
                try:
                    solver.set_parameter(key, value)
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Failed to set parameter {key}={value}: {e}")

            # Solve
            result = solver.solve()

            # Extract results
            status = self._cuopt_status_to_enum(result.status)
            objective = result.objective if hasattr(result, 'objective') else None
            solution = result.x if hasattr(result, 'x') else None

            return SolveResult(
                status=status,
                objective=objective,
                solution=solution,
                iterations=result.iterations if hasattr(result, 'iterations') else 0,
                primal_residual=result.primal_residual if hasattr(result, 'primal_residual') else 0.0,
                dual_residual=result.dual_residual if hasattr(result, 'dual_residual') else 0.0,
                gap=result.gap if hasattr(result, 'gap') else 0.0,
                method_used=str(config.method.name),
                pdlp_mode_used=str(config.pdlp_solver_mode.value),
            )

        except Exception as e:
            logger.error(f"cuOpt solve failed: {e}")
            if self.use_fallback:
                logger.info("Falling back to scipy")
                return self._solve_scipy(lp, config, timeout_ms)
            return SolveResult(
                status=SolveStatus.NUMERICAL_ERROR,
                error_message=str(e),
            )

    def _cuopt_status_to_enum(self, status) -> SolveStatus:
        """Convert cuOpt status to our enum."""
        status_str = str(status).upper()

        if "OPTIMAL" in status_str:
            return SolveStatus.OPTIMAL
        elif "INFEASIBLE" in status_str:
            return SolveStatus.INFEASIBLE
        elif "UNBOUNDED" in status_str:
            return SolveStatus.UNBOUNDED
        elif "TIME" in status_str or "LIMIT" in status_str:
            return SolveStatus.TIMEOUT
        elif "ITERATION" in status_str:
            return SolveStatus.ITERATION_LIMIT
        else:
            return SolveStatus.UNKNOWN

    def _solve_scipy(
        self,
        lp: LPData,
        config: CuOptConfig,
        timeout_ms: float,
    ) -> SolveResult:
        """
        Solve using scipy HiGHS (fallback solver).

        HiGHS supports most LP solving methods and is a good approximation
        for testing parameter tuning logic.
        """
        try:
            from scipy.optimize import linprog, milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
        except ImportError:
            return SolveResult(
                status=SolveStatus.UNKNOWN,
                error_message="scipy not available",
            )

        try:
            lp_dict = lp.to_cuopt_format()
            c = lp_dict["c"]
            A = lp_dict["A"]
            b = lp_dict["b"]
            sense = lp_dict["sense"]
            lb = lp_dict["lb"]
            ub = lp_dict["ub"]

            n_cons = A.shape[0] if A is not None else 0
            n_vars = len(c)

            if n_cons == 0:
                # No constraints - just check bounds
                # Optimal is at lower or upper bounds depending on c sign
                x = np.where(c >= 0, lb, ub)
                x = np.clip(x, lb, ub)
                return SolveResult(
                    status=SolveStatus.OPTIMAL,
                    objective=float(c @ x),
                    solution=x,
                    method_used="scipy_trivial",
                )

            # Separate constraints by type
            eq_mask = sense == 0
            leq_mask = sense == -1
            geq_mask = sense == 1

            # Build scipy constraint matrices
            A_eq = A[eq_mask] if np.any(eq_mask) else None
            b_eq = b[eq_mask] if np.any(eq_mask) else None

            # Combine <= and >= into <= form
            A_ub_parts = []
            b_ub_parts = []

            if np.any(leq_mask):
                A_ub_parts.append(A[leq_mask])
                b_ub_parts.append(b[leq_mask])

            if np.any(geq_mask):
                # >= becomes <= by negation
                A_ub_parts.append(-A[geq_mask])
                b_ub_parts.append(-b[geq_mask])

            if A_ub_parts:
                from scipy.sparse import vstack
                A_ub = vstack(A_ub_parts)
                b_ub = np.concatenate(b_ub_parts)
            else:
                A_ub = None
                b_ub = None

            # Variable bounds
            bounds = list(zip(lb, ub))

            # HiGHS options mapped from cuOpt config
            # HiGHS doesn't have PDLP, but we can use its methods
            highs_method = "highs-ipm"  # interior point
            if config.method == SolverMethod.DUAL_SIMPLEX:
                highs_method = "highs-ds"

            options = {
                "maxiter": min(config.iteration_limit, 1000000),
                "presolve": config.presolve,
                "disp": self.verbose,
                "time_limit": timeout_ms / 1000.0,
            }

            # Add tolerance if supported
            tol = min(
                config.absolute_primal_tolerance,
                config.absolute_dual_tolerance,
                config.absolute_gap_tolerance,
            )

            # Solve
            result = linprog(
                c=c,
                A_ub=A_ub.toarray() if A_ub is not None else None,
                b_ub=b_ub,
                A_eq=A_eq.toarray() if A_eq is not None else None,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options=options,
            )

            # Convert result
            if result.success:
                return SolveResult(
                    status=SolveStatus.OPTIMAL,
                    objective=float(result.fun),
                    solution=result.x,
                    iterations=result.nit if hasattr(result, 'nit') else 0,
                    method_used="scipy_highs",
                )
            else:
                msg = result.message.lower() if hasattr(result, 'message') else ""
                if "infeasible" in msg:
                    status = SolveStatus.INFEASIBLE
                elif "unbounded" in msg:
                    status = SolveStatus.UNBOUNDED
                elif "iteration" in msg:
                    status = SolveStatus.ITERATION_LIMIT
                else:
                    status = SolveStatus.UNKNOWN

                return SolveResult(
                    status=status,
                    error_message=result.message if hasattr(result, 'message') else "Unknown error",
                    method_used="scipy_highs",
                )

        except Exception as e:
            return SolveResult(
                status=SolveStatus.NUMERICAL_ERROR,
                error_message=str(e),
            )


@dataclass
class BatchResult:
    """Result of solving multiple LPs."""
    results: List[SolveResult]
    total_time_ms: float
    throughput_lps_per_sec: float
    success_rate: float
    mean_solve_time_ms: float
    p50_solve_time_ms: float
    p95_solve_time_ms: float
    p99_solve_time_ms: float


def solve_batch(
    lps: List[LPData],
    config: CuOptConfig,
    solver: Optional[CuOptSolver] = None,
    timeout_ms: float = 60000,
) -> BatchResult:
    """
    Solve a batch of LPs with the same configuration.

    Args:
        lps: List of LP problems
        config: cuOpt configuration to use
        solver: Solver instance (creates one if None)
        timeout_ms: Total timeout for batch

    Returns:
        BatchResult with all results and statistics
    """
    if solver is None:
        solver = CuOptSolver()

    if not lps:
        return BatchResult(
            results=[],
            total_time_ms=0,
            throughput_lps_per_sec=0,
            success_rate=0,
            mean_solve_time_ms=0,
            p50_solve_time_ms=0,
            p95_solve_time_ms=0,
            p99_solve_time_ms=0,
        )

    start_time = time.perf_counter()
    results = []
    per_lp_timeout = timeout_ms / len(lps)

    for lp in lps:
        elapsed = (time.perf_counter() - start_time) * 1000
        remaining = timeout_ms - elapsed

        if remaining <= 0:
            # Timeout - fill with timeout status
            results.append(SolveResult(
                status=SolveStatus.TIMEOUT,
                error_message="Batch timeout exceeded",
            ))
            continue

        result = solver.solve(lp, config, min(per_lp_timeout, remaining))
        results.append(result)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Compute statistics
    solve_times = [r.solve_time_ms for r in results if r.solve_time_ms > 0]
    n_success = sum(1 for r in results if r.status == SolveStatus.OPTIMAL)

    if solve_times:
        solve_times_sorted = sorted(solve_times)
        n = len(solve_times_sorted)
        p50_idx = int(n * 0.50)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)

        mean_time = sum(solve_times) / len(solve_times)
        p50_time = solve_times_sorted[p50_idx]
        p95_time = solve_times_sorted[p95_idx]
        p99_time = solve_times_sorted[p99_idx]
    else:
        mean_time = p50_time = p95_time = p99_time = 0.0

    return BatchResult(
        results=results,
        total_time_ms=total_time_ms,
        throughput_lps_per_sec=len(lps) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        success_rate=n_success / len(lps) if lps else 0,
        mean_solve_time_ms=mean_time,
        p50_solve_time_ms=p50_time,
        p95_solve_time_ms=p95_time,
        p99_solve_time_ms=p99_time,
    )
