"""
Quick escalation test for gen2c.py - Level 1 validation
"""
import sys
import time
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
candidate = load_module(".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen2c.py", "gen2c")
parent = load_module(".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen1x.py", "gen1x")
champion = load_module(".evolve-sdk/evolve_fast_n_queens_solvers_t/mutations/gen1b.py", "gen1b")

def verify_solution(n, board):
    """Verify a solution is correct for N-Queens."""
    if board is None:
        return False
    if len(board) != n:
        return False
    if set(board) != set(range(n)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == abs(i - j):
                return False
    return True

results = {"tests_run": [], "failures": [], "warnings": [], "timings": {}}

print("=" * 60)
print("LEVEL 1 ESCALATION: Extended Test Suite for gen2c.py")
print("=" * 60)

# TEST 1: Edge Cases
print("\n[TEST 1] Edge Cases - Small N values")
for n, expected in [(1, True), (2, False), (3, False), (4, True), (5, True), (6, True), (7, True)]:
    results["tests_run"].append(f"edge_n{n}")
    result = candidate.solve_nqueens(n)
    has_solution = result is not None
    if has_solution != expected:
        results["failures"].append(f"edge_n{n}: Expected solvable={expected}, got {has_solution}")
        print(f"  N={n}: FAIL")
    elif has_solution and not verify_solution(n, result):
        results["failures"].append(f"edge_n{n}: Invalid solution")
        print(f"  N={n}: FAIL - Invalid solution")
    else:
        print(f"  N={n}: PASS")

# TEST 2: Standard cases
print("\n[TEST 2] Standard Test Cases")
for n in [8, 12, 16, 20]:
    results["tests_run"].append(f"std_n{n}")
    start = time.perf_counter()
    result = candidate.solve_nqueens(n)
    elapsed = time.perf_counter() - start
    results["timings"][f"candidate_n{n}"] = elapsed
    if not verify_solution(n, result):
        results["failures"].append(f"std_n{n}: Invalid solution")
        print(f"  N={n}: FAIL")
    else:
        print(f"  N={n}: PASS - {elapsed*1000:.2f}ms")

# TEST 3: Out-of-Distribution (smaller set, with timeout)
print("\n[TEST 3] Out-of-Distribution - Larger N")
for n in [24, 28]:
    results["tests_run"].append(f"ood_n{n}")
    start = time.perf_counter()
    result = candidate.solve_nqueens(n)
    elapsed = time.perf_counter() - start
    results["timings"][f"candidate_n{n}"] = elapsed
    if elapsed > 5.0:
        results["failures"].append(f"ood_n{n}: Timeout ({elapsed:.2f}s)")
        print(f"  N={n}: FAIL - Timeout")
    elif not verify_solution(n, result):
        results["failures"].append(f"ood_n{n}: Invalid solution")
        print(f"  N={n}: FAIL - Invalid")
    else:
        print(f"  N={n}: PASS - {elapsed*1000:.2f}ms")

# TEST 4: Comparative Performance
print("\n[TEST 4] Comparative Performance (10 iterations)")
for n in [8, 12, 16, 20]:
    results["tests_run"].append(f"perf_n{n}")

    start = time.perf_counter()
    for _ in range(10): candidate.solve_nqueens(n)
    cand_t = (time.perf_counter() - start) / 10

    start = time.perf_counter()
    for _ in range(10): parent.solve_nqueens(n)
    par_t = (time.perf_counter() - start) / 10

    start = time.perf_counter()
    for _ in range(10): champion.solve_nqueens(n)
    champ_t = (time.perf_counter() - start) / 10

    results["timings"][f"n{n}_cand_avg"] = cand_t
    results["timings"][f"n{n}_par_avg"] = par_t
    results["timings"][f"n{n}_champ_avg"] = champ_t

    imp_vs_par = ((par_t - cand_t) / par_t) * 100
    imp_vs_champ = ((champ_t - cand_t) / champ_t) * 100

    print(f"  N={n}: Cand={cand_t*1000:.3f}ms, Parent={par_t*1000:.3f}ms ({imp_vs_par:+.1f}%), Champion={champ_t*1000:.3f}ms ({imp_vs_champ:+.1f}%)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Tests Run: {len(results['tests_run'])}")
print(f"Failures: {len(results['failures'])}")

if results["failures"]:
    print("\nFAILURES:")
    for f in results["failures"]:
        print(f"  - {f}")

# Calculate total timings for fitness estimate
total_cand = sum(v for k, v in results["timings"].items() if "cand_avg" in k)
total_par = sum(v for k, v in results["timings"].items() if "par_avg" in k)
total_champ = sum(v for k, v in results["timings"].items() if "champ_avg" in k)

print(f"\nAggregate (N=8,12,16,20 avg):")
print(f"  Candidate total: {total_cand*1000:.2f}ms")
print(f"  Parent total: {total_par*1000:.2f}ms")
print(f"  Champion total: {total_champ*1000:.2f}ms")

if total_par > 0:
    overall_imp = ((total_par - total_cand) / total_par) * 100
    print(f"  Candidate vs Parent: {overall_imp:+.1f}%")

print(f"\nVERDICT: {'PASSED' if len(results['failures']) == 0 else 'FAILED'}")
