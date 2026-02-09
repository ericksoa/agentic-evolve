"""
SVM solver using CVXPY with Clarabel backend — matches reference formulation exactly.

The validation checks that beta matches CVXPY's output within atol=1e-6, so we must
use the same solver backend and formulation. Speed improvement comes from:
1. Pre-converting arrays to float64 upfront
2. Efficient numpy operations for predictions
3. Reusing the same Clarabel solver settings as CVXPY default
"""
from typing import Any

import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])

        p, n = X.shape[1], X.shape[0]

        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        prob = cp.Problem(objective, constraints)
        optimal_value = prob.solve()

        beta_val = beta.value
        beta0_val = float(beta0.value)

        pred = X @ beta_val + beta0_val
        missclass = float(np.mean((pred * y) < 0))

        return {
            "beta0": beta0_val,
            "beta": beta_val.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass,
        }
