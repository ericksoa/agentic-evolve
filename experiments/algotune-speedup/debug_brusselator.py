#!/usr/bin/env python3
"""
Diagnostic script to understand Brusselator ODE tolerance sensitivity.

Tests what tolerance levels are needed for solve_ivp RK45 to pass the
validation check: np.allclose(ref, cand, rtol=1e-5, atol=1e-8).

The reference solver uses RK45 with rtol=1e-8, atol=1e-8.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add algotune to path
SHOWCASE_DIR = Path(__file__).resolve().parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
sys.path.insert(0, str(ALGOTUNE_DIR))

from scipy.integrate import solve_ivp
from AlgoTuneTasks.factory import TaskFactory


def solve_brusselator(problem, rtol, atol, method="RK45"):
    """Solve the Brusselator with given tolerances."""
    y0 = np.array(problem["y0"])
    t0, t1 = problem["t0"], problem["t1"]
    params = problem["params"]
    A = params["A"]
    B = params["B"]

    def brusselator(t, y):
        X, Y = y
        dX_dt = A + X**2 * Y - (B + 1) * X
        dY_dt = B * X - X**2 * Y
        return np.array([dX_dt, dY_dt])

    sol = solve_ivp(
        brusselator,
        [t0, t1],
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if sol.success:
        return sol.y[:, -1]
    else:
        return None


def check_allclose(ref, cand, label):
    """Check np.allclose and print diagnostics."""
    val_rtol = 1e-5
    val_atol = 1e-8

    abs_diff = np.abs(ref - cand)
    # Relative diff as computed in is_solution
    rel_diff = np.abs((ref - cand) / (val_atol + val_rtol * np.abs(ref)))
    passes = np.allclose(ref, cand, rtol=val_rtol, atol=val_atol)

    print(f"  {label}:")
    print(f"    ref       = {ref}")
    print(f"    candidate = {cand}")
    print(f"    abs_diff  = {abs_diff}  (max={np.max(abs_diff):.3e})")
    print(f"    rel_diff  = {rel_diff}  (max={np.max(rel_diff):.3e})")
    print(f"    allclose  = {passes}")
    return passes


def main():
    task = TaskFactory("ode_brusselator")

    # Test configurations: (size, seed)
    test_configs = [
        (10, 42),
        (10, 123),
        (25, 42),
        (25, 123),
        (50, 42),
        (50, 123),
        (100, 42),
        (100, 123),
    ]

    # Tolerance levels to test
    tolerance_levels = [
        ("rtol=1e-4, atol=1e-4", 1e-4, 1e-4),
        ("rtol=1e-5, atol=1e-5", 1e-5, 1e-5),
        ("rtol=1e-6, atol=1e-6", 1e-6, 1e-6),
        ("rtol=1e-7, atol=1e-7", 1e-7, 1e-7),
        ("rtol=1e-8, atol=1e-8", 1e-8, 1e-8),
        ("rtol=1e-9, atol=1e-9", 1e-9, 1e-9),
        ("rtol=1e-6, atol=1e-8", 1e-6, 1e-8),
        ("rtol=1e-7, atol=1e-8", 1e-7, 1e-8),
        ("rtol=1e-6, atol=1e-9", 1e-6, 1e-9),
    ]

    # Alternative methods to try at reference tolerance
    alt_methods = [
        ("RK23", 1e-8, 1e-8),
        ("DOP853", 1e-8, 1e-8),
        ("Radau", 1e-8, 1e-8),
        ("BDF", 1e-8, 1e-8),
        ("LSODA", 1e-8, 1e-8),
    ]

    print("=" * 80)
    print("BRUSSELATOR ODE TOLERANCE DIAGNOSTIC")
    print("=" * 80)

    # Summary table
    summary = {}

    for size, seed in test_configs:
        print(f"\n{'='*80}")
        print(f"Problem: size={size}, seed={seed}")
        print(f"{'='*80}")

        problem = task.generate_problem(n=size, random_seed=seed)
        print(f"  A={problem['params']['A']:.6f}, B={problem['params']['B']:.6f}")
        print(f"  y0={problem['y0']}")
        print(f"  t_span=[{problem['t0']}, {problem['t1']}]")

        # Reference solution
        ref = solve_brusselator(problem, 1e-8, 1e-8, "RK45")
        if ref is None:
            print("  *** REFERENCE SOLVER FAILED ***")
            continue
        print(f"\n  Reference (RK45 rtol=1e-8, atol=1e-8): {ref}")

        key = f"n={size},s={seed}"
        summary[key] = {}

        # Test each tolerance level
        print(f"\n  --- Tolerance sweep (RK45) ---")
        for label, rtol, atol in tolerance_levels:
            cand = solve_brusselator(problem, rtol, atol, "RK45")
            if cand is None:
                print(f"  {label}: SOLVER FAILED")
                summary[key][label] = "FAIL(solver)"
                continue
            passes = check_allclose(ref, cand, label)
            summary[key][label] = "PASS" if passes else "FAIL"

        # Test alternative methods at reference tolerance
        print(f"\n  --- Alternative methods (rtol=1e-8, atol=1e-8) ---")
        for method, rtol, atol in alt_methods:
            cand = solve_brusselator(problem, rtol, atol, method)
            if cand is None:
                print(f"  {method}: SOLVER FAILED")
                summary[key][f"{method}@1e-8"] = "FAIL(solver)"
                continue
            passes = check_allclose(ref, cand, f"{method} rtol=1e-8, atol=1e-8")
            summary[key][f"{method}@1e-8"] = "PASS" if passes else "FAIL"

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")

    # Collect all labels
    all_labels = []
    for label, _, _ in tolerance_levels:
        all_labels.append(label)
    for method, _, _ in alt_methods:
        all_labels.append(f"{method}@1e-8")

    # Print header
    print(f"{'Config':<20}", end="")
    for label in all_labels:
        short = label.replace("rtol=", "r").replace(", atol=", " a").replace("1e-", "")
        print(f" {short:>12}", end="")
    print()

    # Print rows
    for key in summary:
        print(f"{key:<20}", end="")
        for label in all_labels:
            val = summary[key].get(label, "?")
            print(f" {val:>12}", end="")
        print()

    # Also test: what if we use a different RK45 implementation altogether?
    # E.g., does the max_step matter?
    print(f"\n\n{'='*80}")
    print("MAX_STEP SENSITIVITY TEST (size=100, seed=42, rtol=1e-6, atol=1e-6)")
    print(f"{'='*80}")

    problem = task.generate_problem(n=100, random_seed=42)
    ref = solve_brusselator(problem, 1e-8, 1e-8, "RK45")

    for max_step_frac in [1.0, 0.5, 0.1, 0.05, 0.01]:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        A = params["A"]
        B = params["B"]

        def brusselator(t, y, A=A, B=B):
            X, Y = y
            return np.array([A + X**2 * Y - (B + 1) * X, B * X - X**2 * Y])

        max_step = (t1 - t0) * max_step_frac
        sol = solve_ivp(
            brusselator, [t0, t1], y0,
            method="RK45", rtol=1e-6, atol=1e-6,
            max_step=max_step,
        )
        if sol.success:
            cand = sol.y[:, -1]
            passes = check_allclose(ref, cand, f"max_step={max_step:.1f} ({max_step_frac})")
        else:
            print(f"  max_step={max_step:.1f}: SOLVER FAILED")


if __name__ == "__main__":
    main()
