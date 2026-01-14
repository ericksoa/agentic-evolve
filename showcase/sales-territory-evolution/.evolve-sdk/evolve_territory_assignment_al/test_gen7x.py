from mutations.gen7x import assign_territories

# Test 1: Empty accounts
result = assign_territories([], 3)
assert result == {0: [], 1: [], 2: []}, f'Empty test failed: {result}'
print('Test 1 passed: empty accounts')

# Test 2: Zero reps
result = assign_territories([{'id': 1, 'lat': 0, 'lon': 0, 'revenue': 100}], 0)
assert result == {}, f'Zero reps test failed: {result}'
print('Test 2 passed: zero reps')

# Test 3: Basic test with 6 accounts, 2 reps
accounts = [
    {'id': 1, 'lat': 0.0, 'lon': 0.0, 'revenue': 100},
    {'id': 2, 'lat': 0.1, 'lon': 0.1, 'revenue': 150},
    {'id': 3, 'lat': 0.2, 'lon': 0.0, 'revenue': 200},
    {'id': 4, 'lat': 10.0, 'lon': 10.0, 'revenue': 120},
    {'id': 5, 'lat': 10.1, 'lon': 10.1, 'revenue': 180},
    {'id': 6, 'lat': 10.2, 'lon': 10.0, 'revenue': 250},
]
result = assign_territories(accounts, 2)
assert len(result) == 2, f'Wrong number of reps: {result}'
all_ids = set()
for rep_id, account_ids in result.items():
    all_ids.update(account_ids)
assert all_ids == {1, 2, 3, 4, 5, 6}, f'Missing accounts: {all_ids}'
print('Test 3 passed: basic 6 accounts, 2 reps')

# Test 4: More reps than accounts
accounts = [
    {'id': 1, 'lat': 0.0, 'lon': 0.0, 'revenue': 100},
    {'id': 2, 'lat': 1.0, 'lon': 1.0, 'revenue': 200},
]
result = assign_territories(accounts, 5)
assert len(result) == 5, f'Wrong number of reps: {result}'
all_ids = set()
for rep_id, account_ids in result.items():
    all_ids.update(account_ids)
assert all_ids == {1, 2}, f'Missing accounts: {all_ids}'
print('Test 4 passed: more reps than accounts')

# Test 5: Single account
result = assign_territories([{'id': 1, 'lat': 0, 'lon': 0, 'revenue': 100}], 3)
assert len(result) == 3, f'Wrong number of reps: {result}'
all_ids = set()
for rep_id, account_ids in result.items():
    all_ids.update(account_ids)
assert all_ids == {1}, f'Account missing: {all_ids}'
print('Test 5 passed: single account')

print('\nAll tests passed!')
