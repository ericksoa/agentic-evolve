"""Verification test for gen1x crossover hybrid."""
import sys, numpy as np
sys.path.insert(0, '.')
import importlib.util

spec = importlib.util.spec_from_file_location(
    'gen1x',
    '.evolve-sdk/optimize_dijkstra_from_indices/mutations/gen1x.py'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Module loaded successfully')
solver = mod.Solver()
print('Solver instantiated')

from scipy.sparse import random as sp_random
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

def check(result, ref, source_indices, n, label):
    mismatch = False
    for i in range(len(source_indices)):
        for j in range(n):
            r = result['distances'][i][j]
            s = float(ref[i][j])
            if s == float('inf'):
                if r is not None:
                    mismatch = True
                    print(f'{label} MISMATCH [{i}][{j}]: expected None got {r}')
            else:
                if r is None:
                    mismatch = True
                    print(f'{label} MISMATCH [{i}][{j}]: expected {s} got None')
                elif abs(r - s) > 1e-5 * abs(s) + 1e-8:
                    mismatch = True
                    print(f'{label} MISMATCH [{i}][{j}]: {s} vs {r}')
    print(f'{label}: {"PASSED" if not mismatch else "FAILED"}')
    return not mismatch

all_pass = True

# Test 1: Small graph
np.random.seed(42)
n = 20
g = sp_random(n, n, density=0.2, format='csr', dtype=np.float64)
g.data[:] = np.random.randint(1, 101, size=g.data.shape).astype(np.float64)
g = g.minimum(g.T).tocsr()
g.setdiag(0)
g.eliminate_zeros()
src = [0, 3, 7]
prob = {'data': g.data.tolist(), 'indices': g.indices.tolist(),
        'indptr': g.indptr.tolist(), 'shape': [n, n], 'source_indices': src}
res = solver.solve(prob)
ref = sp_dijkstra(g, directed=False, indices=src, min_only=False)
all_pass &= check(res, ref, src, n, 'Test1(n=20)')

# Test 2: Larger graph
np.random.seed(123)
n = 100
g = sp_random(n, n, density=0.2, format='csr', dtype=np.float64)
g.data[:] = np.random.randint(1, 101, size=g.data.shape).astype(np.float64)
g = g.minimum(g.T).tocsr()
g.setdiag(0)
g.eliminate_zeros()
src = list(range(0, n, 10))
prob = {'data': g.data.tolist(), 'indices': g.indices.tolist(),
        'indptr': g.indptr.tolist(), 'shape': [n, n], 'source_indices': src}
res = solver.solve(prob)
ref = sp_dijkstra(g, directed=False, indices=src, min_only=False)
all_pass &= check(res, ref, src, n, 'Test2(n=100)')

# Test 3: Empty
res3 = solver.solve({'data': [], 'indices': [], 'indptr': [0],
                      'shape': [1, 1], 'source_indices': []})
t3 = res3 == {'distances': []}
print(f'Test3(empty): {"PASSED" if t3 else "FAILED"}')
all_pass &= t3

# Test 4: Single source
prob4 = {'data': g.data.tolist(), 'indices': g.indices.tolist(),
         'indptr': g.indptr.tolist(), 'shape': [n, n], 'source_indices': [0]}
res4 = solver.solve(prob4)
ref4 = sp_dijkstra(g, directed=False, indices=[0], min_only=False)
all_pass &= check(res4, ref4, [0], n, 'Test4(single)')

print(f'\nOverall: {"ALL PASSED" if all_pass else "SOME FAILED"}')
sys.exit(0 if all_pass else 1)
