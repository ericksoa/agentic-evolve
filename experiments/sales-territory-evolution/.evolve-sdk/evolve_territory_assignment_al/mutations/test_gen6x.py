#!/usr/bin/env python3
"""Quick test for gen6x crossover hybrid."""
import sys
sys.path.insert(0, '.')

# Import the function
exec(open('.evolve-sdk/evolve_territory_assignment_al/mutations/gen6x.py').read())

# Basic test
accounts = [
    {"id": 1, "lat": 0.0, "lon": 0.0, "revenue": 100},
    {"id": 2, "lat": 1.0, "lon": 0.0, "revenue": 200},
    {"id": 3, "lat": 0.0, "lon": 1.0, "revenue": 150},
    {"id": 4, "lat": 1.0, "lon": 1.0, "revenue": 250},
]
result = assign_territories(accounts, 2)
print("Test result:", result)

# Verify structure
assert isinstance(result, dict), "Result should be a dict"
assert len(result) == 2, "Should have 2 territories"
all_ids = []
for rep_id, account_ids in result.items():
    all_ids.extend(account_ids)
assert sorted(all_ids) == [1, 2, 3, 4], f"All accounts should be assigned, got {all_ids}"

# Edge case: empty accounts
result_empty = assign_territories([], 3)
assert result_empty == {0: [], 1: [], 2: []}, "Empty accounts should return empty territories"

# Edge case: single account
result_single = assign_territories([{"id": 1, "lat": 0.0, "lon": 0.0, "revenue": 100}], 2)
assert len(result_single) == 2, "Should have 2 territories even with 1 account"

# Edge case: zero reps
result_zero = assign_territories(accounts, 0)
assert result_zero == {}, "Zero reps should return empty dict"

# Larger test for balancing
import random
import math
random.seed(42)
large_accounts = [
    {"id": i, "lat": random.uniform(0, 100), "lon": random.uniform(0, 100), "revenue": random.uniform(100, 10000)}
    for i in range(100)
]
result_large = assign_territories(large_accounts, 5)
assert len(result_large) == 5, "Should have 5 territories"
all_large_ids = []
for rep_id, account_ids in result_large.items():
    all_large_ids.extend(account_ids)
assert sorted(all_large_ids) == list(range(100)), "All large accounts should be assigned"

# Check revenue balance on large test
total_rev = sum(a["revenue"] for a in large_accounts)
target_rev = total_rev / 5
rep_revs = []
for rep_id, account_ids in result_large.items():
    rev = sum(large_accounts[aid]["revenue"] for aid in account_ids)
    rep_revs.append(rev)
    print(f"  Rep {rep_id}: {len(account_ids)} accounts, revenue={rev:.2f} (target={target_rev:.2f})")

mean_rev = sum(rep_revs) / 5
variance = sum((r - mean_rev) ** 2 for r in rep_revs) / 5
cv = math.sqrt(variance) / mean_rev
print(f"Revenue CV: {cv:.4f}")

print("All tests passed!")
