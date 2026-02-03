"""
Correctness tests for the transaction summarizer.

These tests are THE source of truth. They define exactly what any valid
implementation must do, regardless of how it's structured internally.
The evolution process never sees these tests -- they exist to verify
that the evolved function still computes the right answers.
"""

import importlib.util
import sys
import math
import argparse


def load_function(path):
    """Dynamically load the summarize function from any solution file."""
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find the callable -- we accept any single public function
    candidates = [
        name for name in dir(mod)
        if callable(getattr(mod, name)) and not name.startswith("_")
    ]
    if len(candidates) == 1:
        return getattr(mod, candidates[0])
    # Prefer well-known names
    for name in ("summarize_transactions", "do_stuff", "summarize", "process_transactions"):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Could not find a suitable function in {path}. Found: {candidates}")


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

SIMPLE_SALES = [
    {"type": "sale", "amount": 100.0, "customer": "Alice", "category": "Books"},
    {"type": "sale", "amount": 200.0, "customer": "Bob", "category": "Electronics"},
    {"type": "sale", "amount": 50.0, "customer": "Alice", "category": "Books"},
]

MIXED_TRANSACTIONS = [
    {"type": "sale", "amount": 500.0, "customer": "Alice", "category": "Electronics"},
    {"type": "sale", "amount": 300.0, "customer": "Bob", "category": "Books"},
    {"type": "refund", "amount": 100.0, "customer": "Alice", "category": "Electronics"},
    {"type": "sale", "amount": 200.0, "customer": "Charlie", "category": "Books"},
    {"type": "sale", "amount": 15000.0, "customer": "WhaleInc", "category": "Enterprise"},
]

EMPTY_LIST = []

REFUND_ONLY = [
    {"type": "refund", "amount": 50.0, "customer": "Dave", "category": "Books"},
]

MANY_CUSTOMERS = [
    {"type": "sale", "amount": float(i * 100), "customer": f"Cust_{i}", "category": "General"}
    for i in range(1, 11)
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_simple_sales(fn):
    result = fn(SIMPLE_SALES)
    assert math.isclose(result["total"], 350.0), f"total: expected 350.0, got {result['total']}"
    assert result["count"] == 3, f"count: expected 3, got {result['count']}"
    assert math.isclose(result["avg"], 350.0 / 3), f"avg: expected {350/3:.4f}, got {result['avg']}"
    # Alice spent 150, Bob spent 200
    top = dict(result["top_customers"])
    assert math.isclose(top["Bob"], 200.0), f"Bob spend: expected 200, got {top.get('Bob')}"
    assert math.isclose(top["Alice"], 150.0), f"Alice spend: expected 150, got {top.get('Alice')}"
    # Categories
    assert "Books" in result["categories"]
    assert math.isclose(result["categories"]["Books"]["revenue"], 150.0)
    assert result["categories"]["Books"]["orders"] == 2
    assert math.isclose(result["categories"]["Electronics"]["revenue"], 200.0)
    assert result["categories"]["Electronics"]["orders"] == 1
    assert result["has_whale"] is False


def test_mixed_with_refund(fn):
    result = fn(MIXED_TRANSACTIONS)
    # 500 + 300 - 100 + 200 + 15000 = 15900
    assert math.isclose(result["total"], 15900.0), f"total: expected 15900, got {result['total']}"
    assert result["count"] == 5
    # WhaleInc spent 15000 -> has_whale should be True
    assert result["has_whale"] is True
    top = dict(result["top_customers"])
    assert math.isclose(top["WhaleInc"], 15000.0)
    # Alice: 500 - 100 = 400
    assert math.isclose(top["Alice"], 400.0), f"Alice net: expected 400, got {top.get('Alice')}"


def test_empty_list(fn):
    result = fn(EMPTY_LIST)
    assert math.isclose(result["total"], 0.0)
    assert result["count"] == 0
    assert math.isclose(result["avg"], 0.0)
    assert result["top_customers"] == []
    assert result["categories"] == {}
    assert result["has_whale"] is False


def test_refund_only(fn):
    result = fn(REFUND_ONLY)
    assert math.isclose(result["total"], -50.0), f"total: expected -50, got {result['total']}"
    assert result["count"] == 1


def test_top_customers_limited_to_five(fn):
    result = fn(MANY_CUSTOMERS)
    assert len(result["top_customers"]) == 5, f"top_customers length: expected 5, got {len(result['top_customers'])}"
    # Should be sorted descending by spend
    amounts = [c[1] for c in result["top_customers"]]
    assert amounts == sorted(amounts, reverse=True), "top_customers not sorted descending"


def test_category_averages(fn):
    result = fn(SIMPLE_SALES)
    books = result["categories"]["Books"]
    assert math.isclose(books["avg"], 75.0), f"Books avg: expected 75, got {books['avg']}"


ALL_TESTS = [
    test_simple_sales,
    test_mixed_with_refund,
    test_empty_list,
    test_refund_only,
    test_top_customers_limited_to_five,
    test_category_averages,
]


def run_tests(fn, verbose=False):
    """Run all tests against a function. Returns (passed, total, failures)."""
    passed = 0
    failures = []
    for test in ALL_TESTS:
        try:
            test(fn)
            passed += 1
            if verbose:
                print(f"  PASS: {test.__name__}")
        except (AssertionError, Exception) as e:
            failures.append((test.__name__, str(e)))
            if verbose:
                print(f"  FAIL: {test.__name__}: {e}")
    return passed, len(ALL_TESTS), failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a transaction summarizer implementation")
    parser.add_argument("solution_path", help="Path to the solution .py file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    fn = load_function(args.solution_path)
    passed, total, failures = run_tests(fn, verbose=args.verbose)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
