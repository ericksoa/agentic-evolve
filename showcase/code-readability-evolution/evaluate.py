#!/usr/bin/env python3
"""
Fitness evaluator for the code-readability-evolution showcase.

This evaluator scores solutions on THREE dimensions, combining them into
a single fitness number:

  1. CORRECTNESS (hard gate)  -- must pass all functional tests
  2. READABILITY (60% weight) -- static analysis of code quality
  3. PERFORMANCE (40% weight) -- wall-clock time on a realistic workload

If correctness fails, fitness = 0 (no partial credit for broken code).

Readability scoring breakdown (max 100 points):
  - Descriptive names     (0-25): penalize single-char and cryptic names
  - Docstring quality     (0-15): present, describes params/returns
  - Pythonic constructs   (0-20): direct iteration, comprehensions, etc.
  - Low complexity        (0-15): fewer nested branches = better
  - Clean structure       (0-15): reasonable line lengths, no magic numbers
  - Type hints            (0-10): function signature annotations

Performance scoring (max 100 points):
  - Timed on 10,000-transaction synthetic dataset
  - Baseline (do_stuff) calibrated to ~50 points
  - Faster = higher, with diminishing returns via log scaling
"""

import argparse
import ast
import importlib.util
import json
import math
import re
import sys
import time
import random


# ---------------------------------------------------------------------------
# Dynamic loading
# ---------------------------------------------------------------------------

def load_module(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_function(mod):
    candidates = [
        n for n in dir(mod)
        if callable(getattr(mod, n)) and not n.startswith("_")
    ]
    for name in ("summarize_transactions", "do_stuff", "summarize", "process_transactions"):
        if hasattr(mod, name):
            return getattr(mod, name)
    if len(candidates) == 1:
        return getattr(mod, candidates[0])
    raise RuntimeError(f"Cannot find function. Found: {candidates}")


# ---------------------------------------------------------------------------
# Correctness (pass/fail gate)
# ---------------------------------------------------------------------------

def check_correctness(fn):
    """Run the same test suite used by test_transactions.py."""
    from test_transactions import ALL_TESTS
    for test in ALL_TESTS:
        try:
            test(fn)
        except Exception as e:
            return False, f"{test.__name__}: {e}"
    return True, "all passed"


# ---------------------------------------------------------------------------
# Readability scoring (static analysis)
# ---------------------------------------------------------------------------

SINGLE_CHAR = re.compile(r'^[a-z_]$')  # single-letter names
MAGIC_NUMBER = re.compile(r'(?<![a-zA-Z_])\b\d{2,}\b')  # bare numeric literals >= 10


def score_readability(source_code: str) -> dict:
    tree = ast.parse(source_code)
    scores = {}

    # --- 1. Descriptive names (25 pts) ---
    all_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_names.add(node.name)
            for arg in node.args.args:
                all_names.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            all_names.add(node.id)

    if not all_names:
        scores['names'] = 0
    else:
        bad_names = sum(1 for n in all_names if SINGLE_CHAR.match(n) or len(n) <= 2)
        ratio_good = 1 - (bad_names / len(all_names))
        scores['names'] = int(ratio_good * 25)

    # --- 2. Docstring quality (15 pts) ---
    func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not func_defs:
        scores['docstring'] = 0
    else:
        main_fn = func_defs[0]
        doc = ast.get_docstring(main_fn) or ""
        ds = 0
        if doc:
            ds += 5  # exists
            if len(doc) > 30:
                ds += 3  # substantive
            if any(kw in doc.lower() for kw in ("param", "arg", "return", "args", ":")):
                ds += 4  # describes interface
            if any(kw in doc.lower() for kw in ("example", ">>>", "e.g.")):
                ds += 3  # has example
        scores['docstring'] = min(ds, 15)

    # --- 3. Pythonic constructs (20 pts) ---
    pythonic = 20  # start at max, deduct for anti-patterns
    for node in ast.walk(tree):
        # range(len(...)) is a classic anti-pattern
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'range' and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == 'len'):
            pythonic -= 5
        # Manual append in loop instead of comprehension (heuristic)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'append'):
            pythonic -= 1
    # Bonus for comprehensions
    comps = sum(1 for n in ast.walk(tree)
                if isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)))
    pythonic += min(comps * 2, 6)
    scores['pythonic'] = max(0, min(pythonic, 20))

    # --- 4. Cyclomatic complexity proxy (15 pts) ---
    branches = sum(1 for n in ast.walk(tree)
                   if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                     ast.With, ast.BoolOp)))
    # Nesting depth
    max_depth = _max_nesting(tree)
    complexity_penalty = branches * 0.8 + max_depth * 2
    scores['complexity'] = max(0, int(15 - complexity_penalty))

    # --- 5. Clean structure (15 pts) ---
    lines = source_code.split('\n')
    long_lines = sum(1 for l in lines if len(l) > 88)
    total_lines = len([l for l in lines if l.strip()])
    magic_nums = sum(1 for l in lines
                     if not l.strip().startswith('#')
                     and len(MAGIC_NUMBER.findall(l)) > 0)
    struct = 15
    struct -= min(long_lines, 5)
    struct -= min(magic_nums * 2, 6)
    if total_lines > 60:
        struct -= 2  # overly verbose
    scores['structure'] = max(0, struct)

    # --- 6. Type hints (10 pts) ---
    hints = 0
    for fn_node in func_defs:
        if fn_node.returns is not None:
            hints += 5
        annotated_args = sum(1 for a in fn_node.args.args if a.annotation is not None)
        if annotated_args > 0:
            hints += 5
    scores['type_hints'] = min(hints, 10)

    scores['total'] = sum(scores.values()) - scores.get('total', 0)
    # (avoid double-counting if 'total' was already set)
    scores['total'] = (scores['names'] + scores['docstring'] + scores['pythonic']
                       + scores['complexity'] + scores['structure'] + scores['type_hints'])
    return scores


def _max_nesting(tree, _depth=0):
    """Walk the AST and find the maximum nesting depth of control flow."""
    max_d = _depth
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            max_d = max(max_d, _max_nesting(node, _depth + 1))
        else:
            max_d = max(max_d, _max_nesting(node, _depth))
    return max_d


# ---------------------------------------------------------------------------
# Performance scoring
# ---------------------------------------------------------------------------

def generate_workload(n=10000, seed=42):
    rng = random.Random(seed)
    customers = [f"Customer_{i}" for i in range(200)]
    categories = ["Electronics", "Books", "Clothing", "Food", "Software", "Hardware"]
    txns = []
    for _ in range(n):
        is_refund = rng.random() < 0.1
        txns.append({
            "type": "refund" if is_refund else "sale",
            "amount": round(rng.uniform(5, 500), 2),
            "customer": rng.choice(customers),
            "category": rng.choice(categories),
        })
    return txns


def score_performance(fn, workload, runs=5):
    """Time the function and return a score 0-100."""
    # Warm up
    fn(workload)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(workload)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = sorted(times)[len(times) // 2]

    # Baseline calibration: the original do_stuff takes ~X ms on 10k txns.
    # We use log scaling so 2x faster doesn't mean 2x the score.
    # Target: baseline ~ 50 pts, 2x faster ~ 65 pts, 5x faster ~ 80 pts
    baseline_ms = 15.0  # approximate baseline on typical hardware
    solution_ms = median_time * 1000

    if solution_ms <= 0:
        return 100, median_time

    ratio = baseline_ms / solution_ms
    score = 50 + 25 * math.log2(max(ratio, 0.01))
    score = max(0, min(100, score))

    return int(score), median_time


# ---------------------------------------------------------------------------
# Main: combine scores and emit JSON
# ---------------------------------------------------------------------------

def evaluate(solution_path):
    # Add solution directory to path for imports
    import os
    solution_dir = os.path.dirname(os.path.abspath(solution_path))
    if solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)
    # Also add our own directory for test_transactions import
    our_dir = os.path.dirname(os.path.abspath(__file__))
    if our_dir not in sys.path:
        sys.path.insert(0, our_dir)

    mod = load_module(solution_path)
    fn = find_function(mod)

    # 1. Correctness gate
    correct, detail = check_correctness(fn)
    if not correct:
        return {
            "fitness": 0.0,
            "valid": False,
            "error": detail,
            "correctness": False,
            "readability": {},
            "performance": {},
        }

    # 2. Readability
    with open(solution_path) as f:
        source = f.read()
    readability = score_readability(source)

    # 3. Performance
    workload = generate_workload()
    perf_score, median_time = score_performance(fn, workload)

    # Combine: 60% readability, 40% performance
    fitness = 0.6 * readability['total'] + 0.4 * perf_score

    return {
        "fitness": round(fitness, 2),
        "valid": True,
        "correctness": True,
        "readability": readability,
        "readability_score": readability['total'],
        "performance_score": perf_score,
        "performance_ms": round(median_time * 1000, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", help="Path to solution .py file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = evaluate(args.solution)

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Fitness:     {result['fitness']}")
        print(f"Valid:       {result['valid']}")
        print(f"Correct:     {result['correctness']}")
        if result['valid']:
            print(f"Readability: {result['readability_score']}/100  {result['readability']}")
            print(f"Performance: {result['performance_score']}/100  ({result['performance_ms']:.1f}ms)")
