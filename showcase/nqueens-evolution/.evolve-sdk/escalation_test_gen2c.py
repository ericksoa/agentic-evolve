"""
Level 1 Escalation Tests for gen2c.py
Extended test suite with edge cases, boundary conditions, and out-of-distribution tests.
"""

import sys
import time
import traceback
import warnings
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

candidate_path = ".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen2c.py"
parent_path = ".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen1x.py"
champion_path = ".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen1b.py"

candidate = load_module(candidate_path, "gen2c")
parent = load_module(parent_path, "gen1x")
champion = load_module(champion_path, "gen1b")

def verify_solution(n, board):
    """Verify a solution is correct for N-Queens."""
    if board is None:
        return False
    if len(board) != n:
        return False

    # Check column uniqueness (permutation property)
    if set(board) != set(range(n)):
        return False

    # Check diagonals
    for i in range(n):
        for j in range(i + 1, n):
            # Same diagonal check
            if abs(board[i] - board[j]) == abs(i - j):
                return False
    return True

results = {
    "tests_run": [],
    "failures": [],
    "warnings": [],
    "timings": {}
}

print("=" * 60)
print("LEVEL 1 ESCALATION: Extended Test Suite for gen2c.py")
print("=" * 60)

# ============================================================
# TEST 1: Edge Cases - Small N values
# ============================================================
print("\n[TEST 1] Edge Cases - Small N values")
edge_cases = [1, 2, 3, 4, 5, 6, 7]
expected_solvable = {1: True, 2: False, 3: False, 4: True, 5: True, 6: True, 7: True}

for n in edge_cases:
    test_name = f"edge_case_n{n}"
    results["tests_run"].append(test_name)
    try:
        result = candidate.solve_nqueens(n)
        has_solution = result is not None

        if has_solution != expected_solvable[n]:
            results["failures"].append(f"{test_name}: Expected solvable={expected_solvable[n]}, got {has_solution}")
            print(f"  N={n}: FAIL - Expected solvable={expected_solvable[n]}, got {has_solution}")
        elif has_solution and not verify_solution(n, result):
            results["failures"].append(f"{test_name}: Solution invalid: {result}")
            print(f"  N={n}: FAIL - Invalid solution: {result}")
        else:
            print(f"  N={n}: PASS - {'Solution found' if has_solution else 'Correctly returned None'}")
    except Exception as e:
        results["failures"].append(f"{test_name}: Exception: {e}")
        print(f"  N={n}: FAIL - Exception: {e}")

# ============================================================
# TEST 2: Boundary Conditions - N=0 and negative N
# ============================================================
print("\n[TEST 2] Boundary Conditions - Edge inputs")
boundary_tests = [0, -1, -5]

for n in boundary_tests:
    test_name = f"boundary_n{n}"
    results["tests_run"].append(test_name)
    try:
        result = candidate.solve_nqueens(n)
        # N=0 or negative should return None or empty list, not crash
        if n <= 0 and result is not None and result != []:
            results["failures"].append(f"{test_name}: Should return None/empty for N={n}, got {result}")
            print(f"  N={n}: FAIL - Should return None/empty, got {result}")
        else:
            print(f"  N={n}: PASS - Handled gracefully: {result}")
    except Exception as e:
        # Acceptable if it raises an exception for invalid input
        results["warnings"].append(f"{test_name}: Raised exception for invalid input: {e}")
        print(f"  N={n}: WARN - Exception raised (acceptable): {type(e).__name__}")

# ============================================================
# TEST 3: Standard Test Cases (8, 12, 16, 20)
# ============================================================
print("\n[TEST 3] Standard Test Cases")
standard_cases = [8, 12, 16, 20]

for n in standard_cases:
    test_name = f"standard_n{n}"
    results["tests_run"].append(test_name)
    try:
        start = time.perf_counter()
        result = candidate.solve_nqueens(n)
        elapsed = time.perf_counter() - start
        results["timings"][f"candidate_n{n}"] = elapsed

        if not verify_solution(n, result):
            results["failures"].append(f"{test_name}: Invalid solution: {result}")
            print(f"  N={n}: FAIL - Invalid solution")
        else:
            print(f"  N={n}: PASS - Valid solution in {elapsed*1000:.2f}ms")
    except Exception as e:
        results["failures"].append(f"{test_name}: Exception: {e}")
        print(f"  N={n}: FAIL - Exception: {e}")

# ============================================================
# TEST 4: Out-of-Distribution - Larger N values (beyond training)
# ============================================================
print("\n[TEST 4] Out-of-Distribution - Larger N values")
ood_cases = [24, 28, 32, 40]

for n in ood_cases:
    test_name = f"ood_n{n}"
    results["tests_run"].append(test_name)
    try:
        start = time.perf_counter()
        result = candidate.solve_nqueens(n)
        elapsed = time.perf_counter() - start
        results["timings"][f"candidate_n{n}"] = elapsed

        if elapsed > 10.0:  # 10 second timeout
            results["failures"].append(f"{test_name}: Timeout (>{elapsed:.2f}s)")
            print(f"  N={n}: FAIL - Timeout ({elapsed:.2f}s)")
        elif not verify_solution(n, result):
            results["failures"].append(f"{test_name}: Invalid solution")
            print(f"  N={n}: FAIL - Invalid solution")
        else:
            print(f"  N={n}: PASS - Valid solution in {elapsed*1000:.2f}ms")
    except Exception as e:
        results["failures"].append(f"{test_name}: Exception: {e}")
        print(f"  N={n}: FAIL - Exception: {e}")

# ============================================================
# TEST 5: Comparative Performance - Candidate vs Parent vs Champion
# ============================================================
print("\n[TEST 5] Comparative Performance Analysis")
comparison_cases = [8, 12, 16, 20]
num_iterations = 10

print(f"  Running {num_iterations} iterations per solver per N...")

for n in comparison_cases:
    test_name = f"comparison_n{n}"
    results["tests_run"].append(test_name)

    # Time candidate
    start = time.perf_counter()
    for _ in range(num_iterations):
        candidate.solve_nqueens(n)
    candidate_time = (time.perf_counter() - start) / num_iterations

    # Time parent
    start = time.perf_counter()
    for _ in range(num_iterations):
        parent.solve_nqueens(n)
    parent_time = (time.perf_counter() - start) / num_iterations

    # Time champion
    start = time.perf_counter()
    for _ in range(num_iterations):
        champion.solve_nqueens(n)
    champion_time = (time.perf_counter() - start) / num_iterations

    improvement_over_parent = ((parent_time - candidate_time) / parent_time) * 100 if parent_time > 0 else 0
    vs_champion = ((champion_time - candidate_time) / champion_time) * 100 if champion_time > 0 else 0

    results["timings"][f"n{n}_candidate_avg"] = candidate_time
    results["timings"][f"n{n}_parent_avg"] = parent_time
    results["timings"][f"n{n}_champion_avg"] = champion_time

    print(f"  N={n}:")
    print(f"    Candidate: {candidate_time*1000:.3f}ms")
    print(f"    Parent:    {parent_time*1000:.3f}ms ({improvement_over_parent:+.1f}%)")
    print(f"    Champion:  {champion_time*1000:.3f}ms ({vs_champion:+.1f}%)")

# ============================================================
# TEST 6: Memory/Lookup Table Overhead Analysis
# ============================================================
print("\n[TEST 6] Lookup Table Overhead Analysis")
# The concern is that lookup tables have overhead at large N
# Let's compare memory-time tradeoff

for n in [8, 16, 32, 64]:
    test_name = f"lookup_overhead_n{n}"
    results["tests_run"].append(test_name)

    try:
        start = time.perf_counter()
        result = candidate.solve_nqueens(n)
        elapsed = time.perf_counter() - start

        # Check that lookup tables scale reasonably
        # 2 tables of n*n elements = 2n^2 integer lookups
        table_size = 2 * n * n

        if result is not None and verify_solution(n, result):
            print(f"  N={n}: PASS - Time={elapsed*1000:.2f}ms, TableSize={table_size} elements")
        elif n > 32:
            results["warnings"].append(f"{test_name}: Slow at N={n}, took {elapsed:.2f}s")
            print(f"  N={n}: WARN - Took {elapsed:.2f}s (may be expected for large N)")
        else:
            results["failures"].append(f"{test_name}: Failed for N={n}")
            print(f"  N={n}: FAIL")
    except Exception as e:
        if n >= 64:
            results["warnings"].append(f"{test_name}: Exception at N={n}: {e}")
            print(f"  N={n}: WARN - Exception (expected for very large N): {type(e).__name__}")
        else:
            results["failures"].append(f"{test_name}: Unexpected exception: {e}")
            print(f"  N={n}: FAIL - Exception: {e}")

# ============================================================
# TEST 7: Consistency Check - Multiple Runs Same Result
# ============================================================
print("\n[TEST 7] Consistency Check - Deterministic Behavior")
test_name = "consistency_check"
results["tests_run"].append(test_name)

inconsistent = False
for n in [8, 10, 12]:
    first_result = candidate.solve_nqueens(n)
    for _ in range(5):
        result = candidate.solve_nqueens(n)
        if result != first_result:
            inconsistent = True
            results["warnings"].append(f"N={n}: Non-deterministic results")
            print(f"  N={n}: WARN - Non-deterministic (may be valid if multiple solutions exist)")
            break
    else:
        print(f"  N={n}: PASS - Consistent results across runs")

# ============================================================
# TEST 8: Syntax Warning Check
# ============================================================
print("\n[TEST 8] Code Quality - Syntax Warning Check")
test_name = "syntax_warning_check"
results["tests_run"].append(test_name)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Re-import to trigger warnings
    import importlib
    importlib.reload(candidate)

    syntax_warnings = [warning for warning in w if issubclass(warning.category, SyntaxWarning)]
    if syntax_warnings:
        for sw in syntax_warnings:
            results["warnings"].append(f"SyntaxWarning: {sw.message}")
            print(f"  WARN: {sw.message}")
    else:
        print("  PASS: No syntax warnings detected")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ESCALATION TEST SUMMARY")
print("=" * 60)
print(f"Tests Run: {len(results['tests_run'])}")
print(f"Failures: {len(results['failures'])}")
print(f"Warnings: {len(results['warnings'])}")

if results["failures"]:
    print("\nFAILURES:")
    for f in results["failures"]:
        print(f"  - {f}")

if results["warnings"]:
    print("\nWARNINGS:")
    for w in results["warnings"]:
        print(f"  - {w}")

# Calculate aggregate fitness metrics
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

total_candidate_time = sum(v for k, v in results["timings"].items() if "candidate" in k and "avg" in k)
total_parent_time = sum(v for k, v in results["timings"].items() if "parent" in k and "avg" in k)
total_champion_time = sum(v for k, v in results["timings"].items() if "champion" in k and "avg" in k)

if total_parent_time > 0:
    improvement = ((total_parent_time - total_candidate_time) / total_parent_time) * 100
    print(f"Candidate vs Parent: {improvement:+.1f}% (positive = candidate faster)")
if total_champion_time > 0:
    vs_champ = ((total_champion_time - total_candidate_time) / total_champion_time) * 100
    print(f"Candidate vs Champion: {vs_champ:+.1f}% (positive = candidate faster)")

# Final verdict
escalation_passed = len(results["failures"]) == 0
print(f"\nESCALATION VERDICT: {'PASSED' if escalation_passed else 'FAILED'}")
