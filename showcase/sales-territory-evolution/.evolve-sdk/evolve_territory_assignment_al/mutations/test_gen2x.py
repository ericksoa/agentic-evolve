#!/usr/bin/env python3
"""Quick test for gen2x crossover hybrid."""
import sys
sys.path.insert(0, '.')

# Import the function
exec(open('.evolve-sdk/evolve_territory_assignment_al/mutations/gen2x.py').read())

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

print("All tests passed!")
