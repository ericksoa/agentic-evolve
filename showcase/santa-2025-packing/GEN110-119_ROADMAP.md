# Gen110-119 Roadmap: Frontier Optimization Campaign

## Overview

A systematic 10-generation campaign to close the 25% gap to the leaderboard using frontier optimization techniques. Each generation focuses on one core technique, with clear success criteria and fallback plans.

**Starting point**: 86.13 (Gen103)
**Target**: <70 (match leader)
**Timeline**: ~2-3 hours per generation

---

## Gen110: NFP Foundation

### Objective
Build the No-Fit Polygon infrastructure that underpins all serious packing solvers.

### Theory
The NFP of polygons A and B defines all positions where B touches but doesn't overlap A. It's computed via Minkowski sum: `NFP(A,B) = A ⊕ (-B)`.

### Implementation

```python
# python/nfp_core.py
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate, scale
import numpy as np

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

def get_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create a Shapely polygon for a tree at given position and rotation."""
    base = Polygon(TREE_VERTICES)
    rotated = rotate(base, angle_deg, origin=(0, 0))
    return translate(rotated, x, y)

def minkowski_sum_convex(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """Minkowski sum for convex polygons (fast)."""
    coords_a = np.array(poly_a.exterior.coords[:-1])
    coords_b = np.array(poly_b.exterior.coords[:-1])

    # Sort by angle
    def sort_by_angle(coords):
        center = coords.mean(axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        return coords[np.argsort(angles)]

    coords_a = sort_by_angle(coords_a)
    coords_b = sort_by_angle(coords_b)

    # Merge edges by angle (convex Minkowski sum algorithm)
    result = []
    i, j = 0, 0
    n_a, n_b = len(coords_a), len(coords_b)

    while i < n_a or j < n_b:
        result.append(coords_a[i % n_a] + coords_b[j % n_b])

        # Compute edge angles
        edge_a = coords_a[(i + 1) % n_a] - coords_a[i % n_a]
        edge_b = coords_b[(j + 1) % n_b] - coords_b[j % n_b]
        angle_a = np.arctan2(edge_a[1], edge_a[0])
        angle_b = np.arctan2(edge_b[1], edge_b[0])

        if i >= n_a:
            j += 1
        elif j >= n_b:
            i += 1
        elif angle_a < angle_b:
            i += 1
        else:
            j += 1

    return Polygon(result)

def compute_nfp(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """
    Compute No-Fit Polygon.

    NFP = A ⊕ (-B) where -B is B reflected through origin.
    """
    # Reflect B through origin
    reflected_b = scale(poly_b, -1, -1, origin=(0, 0))

    # For non-convex polygons, use decomposition or approximation
    if poly_a.is_valid and reflected_b.is_valid:
        # Approximate with buffering for non-convex
        try:
            return minkowski_sum_convex(poly_a.convex_hull, reflected_b.convex_hull)
        except:
            # Fallback: use buffer approximation
            return poly_a.buffer(0.5).union(reflected_b.buffer(0.5))

    return None

def precompute_nfp_table(angle_steps: int = 8) -> dict:
    """
    Precompute NFP for all rotation pairs.

    Returns:
        nfp_table[angle_a][angle_b] = NFP polygon
    """
    angles = [i * 360.0 / angle_steps for i in range(angle_steps)]
    nfp_table = {}

    for i, angle_a in enumerate(angles):
        nfp_table[i] = {}
        poly_a = get_tree_polygon(0, 0, angle_a)

        for j, angle_b in enumerate(angles):
            poly_b = get_tree_polygon(0, 0, angle_b)
            nfp_table[i][j] = compute_nfp(poly_a, poly_b)

    return nfp_table

def find_valid_placements_nfp(existing: list, new_angle_idx: int,
                               nfp_table: dict, resolution: int = 100) -> list:
    """
    Find all valid placement positions using NFP.

    Returns list of (x, y) positions on the boundary of feasible region.
    """
    if not existing:
        return [(0.0, 0.0)]

    from shapely.ops import unary_union

    # Compute forbidden region (union of all NFPs)
    forbidden_regions = []
    for tree in existing:
        angle_idx = int(tree['angle'] / 45) % 8
        nfp = nfp_table[angle_idx][new_angle_idx]
        # Translate NFP to tree position
        shifted_nfp = translate(nfp, tree['x'], tree['y'])
        forbidden_regions.append(shifted_nfp)

    total_forbidden = unary_union(forbidden_regions)

    # Sample points on boundary of forbidden region
    # These are the "touching" positions
    boundary = total_forbidden.boundary
    valid_positions = []

    for i in range(resolution):
        t = i / resolution
        point = boundary.interpolate(t, normalized=True)
        valid_positions.append((point.x, point.y))

    return valid_positions
```

### Test Plan
```python
def test_nfp_placement():
    """Compare NFP-based placement to Rust greedy for n=10."""
    nfp_table = precompute_nfp_table()

    # NFP-based greedy
    nfp_trees = []
    for i in range(10):
        best_pos = None
        best_side = float('inf')

        for angle_idx in range(8):
            positions = find_valid_placements_nfp(nfp_trees, angle_idx, nfp_table)

            for x, y in positions:
                candidate = nfp_trees + [{'x': x, 'y': y, 'angle': angle_idx * 45}]
                side = compute_side_length(candidate)
                if side < best_side:
                    best_side = side
                    best_pos = (x, y, angle_idx * 45)

        nfp_trees.append({'x': best_pos[0], 'y': best_pos[1], 'angle': best_pos[2]})

    return compute_side_length(nfp_trees)
```

### Success Criteria
- [ ] NFP computation runs in <1s per pair
- [ ] NFP placement matches or beats Rust for n=10
- [ ] Precomputed table ready for use in later generations

### Deliverables
- `python/nfp_core.py` - Core NFP functions
- `python/nfp_table.pkl` - Precomputed NFP table (8x8 rotation pairs)

---

## Gen111: CMA-ES Global Optimization

### Objective
Use CMA-ES to globally optimize tree positions, escaping local optima that SA cannot.

### Theory
CMA-ES maintains a multivariate Gaussian distribution over solutions, adapting both the mean and covariance matrix based on successful samples. It's particularly effective for:
- Non-separable objectives (tree positions are interdependent)
- Rugged landscapes with many local optima
- Problems where gradient information is unavailable

### Implementation

```python
# python/cmaes_optimizer.py
import cma
import numpy as np
from typing import List, Tuple

def trees_to_params(trees: List[dict]) -> np.ndarray:
    """Convert tree list to flat parameter array."""
    params = []
    for t in trees:
        params.extend([t['x'], t['y'], t['angle'] / 360.0])  # Normalize angle
    return np.array(params)

def params_to_trees(params: np.ndarray, n: int) -> List[dict]:
    """Convert flat parameters back to tree list."""
    trees = []
    for i in range(n):
        trees.append({
            'x': params[3*i],
            'y': params[3*i + 1],
            'angle': (params[3*i + 2] % 1.0) * 360.0  # Wrap angle
        })
    return trees

def overlap_penalty(trees: List[dict]) -> float:
    """Compute soft overlap penalty for constraint handling."""
    penalty = 0.0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            overlap = compute_overlap_area(trees[i], trees[j])
            penalty += overlap * 100.0
    return penalty

def cmaes_optimize(n: int, initial_trees: List[dict] = None,
                   max_evals: int = 10000, sigma0: float = 0.3) -> Tuple[float, List[dict]]:
    """
    Optimize packing of n trees using CMA-ES.

    Args:
        n: Number of trees
        initial_trees: Starting solution (or None for random)
        max_evals: Maximum function evaluations
        sigma0: Initial step size

    Returns:
        (best_side_length, best_trees)
    """
    dim = 3 * n

    # Initialize
    if initial_trees:
        x0 = trees_to_params(initial_trees)
    else:
        x0 = np.random.randn(dim) * 0.5

    def objective(params):
        trees = params_to_trees(params, n)

        # Check overlaps
        penalty = overlap_penalty(trees)
        if penalty > 0:
            return 1000.0 + penalty  # Infeasible

        return compute_side_length(trees)

    # Configure CMA-ES
    opts = {
        'maxfevals': max_evals,
        'popsize': 4 + int(3 * np.log(dim)),
        'bounds': [[-15]*dim, [15]*dim],
        'tolfun': 1e-6,
        'verb_disp': 100,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # Run optimization
    while not es.stop():
        solutions = es.ask()
        fitness = [objective(s) for s in solutions]
        es.tell(solutions, fitness)

    # Extract best
    best_params = es.result.xbest
    best_trees = params_to_trees(best_params, n)

    # Local repair if infeasible
    if overlap_penalty(best_trees) > 0:
        best_trees = repair_overlaps(best_trees)

    return compute_side_length(best_trees), best_trees

def repair_overlaps(trees: List[dict], max_iters: int = 1000) -> List[dict]:
    """
    Repair overlapping configuration with local moves.
    """
    for _ in range(max_iters):
        if overlap_penalty(trees) == 0:
            break

        # Find overlapping pair
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if polygons_overlap(trees[i], trees[j]):
                    # Push apart
                    dx = trees[j]['x'] - trees[i]['x']
                    dy = trees[j]['y'] - trees[i]['y']
                    dist = np.sqrt(dx*dx + dy*dy) + 0.001

                    trees[j]['x'] += dx / dist * 0.05
                    trees[j]['y'] += dy / dist * 0.05
                    break

    return trees
```

### Test Plan
```python
def test_cmaes_improvement():
    """Test CMA-ES improvement over greedy baseline."""
    results = []

    for n in [10, 20, 30, 50]:
        # Get greedy baseline
        baseline_trees = rust_greedy_pack(n)
        baseline_side = compute_side_length(baseline_trees)

        # Run CMA-ES starting from baseline
        cmaes_side, cmaes_trees = cmaes_optimize(n, baseline_trees, max_evals=5000)

        improvement = (baseline_side - cmaes_side) / baseline_side * 100
        results.append((n, baseline_side, cmaes_side, improvement))
        print(f"n={n}: {baseline_side:.4f} -> {cmaes_side:.4f} ({improvement:+.2f}%)")

    return results
```

### Success Criteria
- [ ] CMA-ES improves >50% of test cases
- [ ] Average improvement >1% over greedy
- [ ] No regressions (never worse than input)

### Deliverables
- `python/cmaes_optimizer.py` - CMA-ES optimization module
- Benchmark results for n=10,20,30,50,100

---

## Gen112: ILP Optimal Small n

### Objective
Find provably optimal solutions for n=1..15 using Integer Linear Programming.

### Theory
Discretize the problem:
- Positions on a fine grid (e.g., 200x200)
- Rotations at fixed angles (8 options: 0°, 45°, ..., 315°)
- Use NFP lookup tables for overlap constraints

### Implementation

```python
# python/ilp_optimal.py
from ortools.sat.python import cp_model
import numpy as np
from nfp_core import precompute_nfp_table

def ilp_optimal_pack(n: int, grid_res: int = 200, angle_steps: int = 8,
                     timeout: int = 300) -> Tuple[float, List[dict]]:
    """
    Find optimal packing using CP-SAT.

    Args:
        n: Number of trees
        grid_res: Grid resolution for position discretization
        angle_steps: Number of rotation options
        timeout: Solver timeout in seconds

    Returns:
        (side_length, trees) or (None, None) if infeasible
    """
    model = cp_model.CpModel()

    # Precompute tree bounds for each rotation
    tree_bounds = compute_tree_bounds_all_rotations(angle_steps)

    # Precompute NFP constraints (discretized)
    nfp_constraints = precompute_nfp_constraints(angle_steps, grid_res)

    # Variables
    x = [model.NewIntVar(0, grid_res, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, grid_res, f'y_{i}') for i in range(n)]
    r = [model.NewIntVar(0, angle_steps - 1, f'r_{i}') for i in range(n)]
    side = model.NewIntVar(1, grid_res, 'side')

    # Containment constraints: all vertices within [0, side]
    for i in range(n):
        for ri in range(angle_steps):
            bounds = tree_bounds[ri]  # (min_x, min_y, max_x, max_y) in grid units

            # If r[i] == ri, then position + bounds must be within [0, side]
            b_ri = model.NewBoolVar(f'b_{i}_{ri}')
            model.Add(r[i] == ri).OnlyEnforceIf(b_ri)
            model.Add(r[i] != ri).OnlyEnforceIf(b_ri.Not())

            # x[i] + max_x <= side when r[i] == ri
            model.Add(x[i] + bounds['max_x'] <= side).OnlyEnforceIf(b_ri)
            model.Add(x[i] + bounds['min_x'] >= 0).OnlyEnforceIf(b_ri)
            model.Add(y[i] + bounds['max_y'] <= side).OnlyEnforceIf(b_ri)
            model.Add(y[i] + bounds['min_y'] >= 0).OnlyEnforceIf(b_ri)

    # Non-overlap constraints using NFP
    for i in range(n):
        for j in range(i + 1, n):
            # For each rotation pair, position j must be outside NFP of i
            for ri in range(angle_steps):
                for rj in range(angle_steps):
                    add_nfp_constraint(model, x[i], y[i], x[j], y[j],
                                      r[i], ri, r[j], rj,
                                      nfp_constraints[ri][rj])

    # Symmetry breaking: first tree at origin, second tree in first quadrant
    model.Add(x[0] == 0)
    model.Add(y[0] == 0)
    if n > 1:
        model.Add(x[1] >= x[0])
        model.Add(y[1] >= y[0])

    # Objective: minimize side
    model.Minimize(side)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 8  # Parallel search

    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        trees = []
        scale = 10.0 / grid_res  # Convert grid to actual coordinates

        for i in range(n):
            trees.append({
                'x': solver.Value(x[i]) * scale,
                'y': solver.Value(y[i]) * scale,
                'angle': solver.Value(r[i]) * (360.0 / angle_steps)
            })

        actual_side = compute_side_length(trees)
        return actual_side, trees

    return None, None

def add_nfp_constraint(model, x_i, y_i, x_j, y_j, r_i, ri_val, r_j, rj_val, nfp_poly):
    """
    Add constraint: if r_i == ri_val and r_j == rj_val,
    then (x_j - x_i, y_j - y_i) must be outside nfp_poly.

    Approximated using linear inequalities from NFP vertices.
    """
    # Create indicator for this rotation pair
    b = model.NewBoolVar(f'nfp_{ri_val}_{rj_val}')
    model.Add(r_i == ri_val).OnlyEnforceIf(b)
    model.Add(r_j == rj_val).OnlyEnforceIf(b)

    # NFP gives forbidden relative positions
    # At least one separating constraint must hold
    dx = model.NewIntVar(-1000, 1000, 'dx')
    dy = model.NewIntVar(-1000, 1000, 'dy')
    model.Add(dx == x_j - x_i)
    model.Add(dy == y_j - y_i)

    # Add disjunctive constraints from NFP boundary
    # (Simplified: use bounding box of NFP)
    nfp_bounds = get_nfp_bounds(nfp_poly)

    sep_vars = []
    for edge in get_nfp_edges(nfp_poly):
        # Each edge gives a half-plane constraint
        sv = model.NewBoolVar('sep')
        sep_vars.append(sv)
        # a*dx + b*dy >= c  when sv is true
        model.Add(edge['a'] * dx + edge['b'] * dy >= edge['c']).OnlyEnforceIf(sv)

    # At least one separation constraint must hold when rotation pair matches
    model.AddBoolOr(sep_vars).OnlyEnforceIf(b)
```

### Test Plan
```python
def test_ilp_optimality():
    """Verify ILP finds better solutions than heuristics for small n."""
    results = []

    for n in range(1, 16):
        # Get heuristic solution
        heuristic_trees = rust_greedy_pack(n)
        heuristic_side = compute_side_length(heuristic_trees)

        # Run ILP (5 min timeout)
        ilp_side, ilp_trees = ilp_optimal_pack(n, timeout=300)

        if ilp_side:
            improvement = (heuristic_side - ilp_side) / heuristic_side * 100
            status = "OPTIMAL" if ilp_side < heuristic_side - 0.001 else "MATCH"
        else:
            improvement = 0
            status = "TIMEOUT"

        results.append((n, heuristic_side, ilp_side, improvement, status))
        print(f"n={n}: heuristic={heuristic_side:.4f}, ILP={ilp_side:.4f} ({status})")

    return results
```

### Success Criteria
- [ ] ILP finds solution for all n=1..10 within timeout
- [ ] ILP beats heuristic for at least 5 of n=1..10
- [ ] At least one provably optimal solution found

### Deliverables
- `python/ilp_optimal.py` - ILP solver module
- Optimal solutions for n=1..15 (where found)

---

## Gen113: Differentiable Packing

### Objective
Make packing differentiable and use gradient descent for optimization.

### Theory
Replace hard overlap constraints with soft, differentiable penalties:
- Signed distance functions (SDF)
- Differentiable polygon intersection
- Smooth approximations to max/min

### Implementation

```python
# python/differentiable_packing.py
import torch
import torch.nn as nn
import numpy as np

class DifferentiablePacking(nn.Module):
    def __init__(self, n: int, tree_vertices: torch.Tensor):
        super().__init__()
        self.n = n
        self.tree_vertices = tree_vertices  # (15, 2)

        # Learnable parameters
        self.positions = nn.Parameter(torch.randn(n, 2) * 0.5)
        self.angles = nn.Parameter(torch.rand(n) * 2 * np.pi)

    def get_transformed_vertices(self) -> torch.Tensor:
        """
        Transform tree vertices by position and rotation.

        Returns: (n, 15, 2) tensor
        """
        cos_a = torch.cos(self.angles)  # (n,)
        sin_a = torch.sin(self.angles)  # (n,)

        # Rotation matrices (n, 2, 2)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a], dim=1),
            torch.stack([sin_a, cos_a], dim=1)
        ], dim=1)

        # Rotate vertices: (n, 15, 2)
        rotated = torch.einsum('nij,kj->nki', rot, self.tree_vertices)

        # Translate
        translated = rotated + self.positions.unsqueeze(1)

        return translated

    def compute_bounding_box(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute soft bounding box side length.

        Uses smooth max/min for differentiability.
        """
        # Flatten to (n*15, 2)
        flat = vertices.view(-1, 2)

        # Smooth max/min using logsumexp
        temperature = 0.1
        max_x = temperature * torch.logsumexp(flat[:, 0] / temperature, dim=0)
        max_y = temperature * torch.logsumexp(flat[:, 1] / temperature, dim=0)
        min_x = -temperature * torch.logsumexp(-flat[:, 0] / temperature, dim=0)
        min_y = -temperature * torch.logsumexp(-flat[:, 1] / temperature, dim=0)

        width = max_x - min_x
        height = max_y - min_y

        # Soft max of width and height
        side = temperature * torch.logsumexp(
            torch.stack([width, height]) / temperature, dim=0
        )

        return side

    def compute_overlap_penalty(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Differentiable overlap penalty using signed distance.
        """
        penalty = torch.tensor(0.0)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Approximate overlap using centroid distance
                centroid_i = vertices[i].mean(dim=0)
                centroid_j = vertices[j].mean(dim=0)

                dist = torch.norm(centroid_i - centroid_j)

                # Approximate collision distance
                collision_dist = 0.6  # Approximate tree diameter

                # Soft penalty: exponential for close trees
                if dist < collision_dist * 2:
                    overlap = torch.relu(collision_dist - dist)
                    penalty = penalty + overlap ** 2

        return penalty

    def forward(self) -> torch.Tensor:
        vertices = self.get_transformed_vertices()

        side = self.compute_bounding_box(vertices)
        overlap_penalty = self.compute_overlap_penalty(vertices)

        # Combined loss
        loss = side + 100.0 * overlap_penalty

        return loss

    def get_trees(self) -> list:
        """Extract tree configuration."""
        return [
            {
                'x': self.positions[i, 0].item(),
                'y': self.positions[i, 1].item(),
                'angle': (self.angles[i].item() * 180 / np.pi) % 360
            }
            for i in range(self.n)
        ]

def optimize_differentiable(n: int, initial_trees: list = None,
                           steps: int = 5000, lr: float = 0.01) -> tuple:
    """
    Optimize packing using gradient descent.
    """
    tree_vertices = torch.tensor(TREE_VERTICES, dtype=torch.float32)

    model = DifferentiablePacking(n, tree_vertices)

    # Initialize from provided solution
    if initial_trees:
        with torch.no_grad():
            for i, t in enumerate(initial_trees):
                model.positions[i] = torch.tensor([t['x'], t['y']])
                model.angles[i] = torch.tensor(t['angle'] * np.pi / 180)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    best_loss = float('inf')
    best_trees = None

    for step in range(steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            current_trees = model.get_trees()
            if not has_any_overlap(current_trees):
                side = compute_side_length(current_trees)
                if side < best_loss:
                    best_loss = side
                    best_trees = current_trees

    return best_loss, best_trees
```

### Success Criteria
- [ ] Gradient descent converges for n=10, 20, 30
- [ ] Finds feasible solutions (no overlaps)
- [ ] Competitive with CMA-ES on at least 30% of cases

### Deliverables
- `python/differentiable_packing.py` - Differentiable optimization module

---

## Gen114: GPU Batch Evaluation

### Objective
Massively parallel evaluation of packing candidates on GPU.

### Implementation

```python
# python/gpu_batch_eval.py
import torch
import torch.nn.functional as F

class GPUPackingEvaluator:
    def __init__(self, tree_vertices: torch.Tensor, device: str = 'mps'):
        """
        Args:
            tree_vertices: (15, 2) base tree polygon
            device: 'cuda', 'mps' (Apple Silicon), or 'cpu'
        """
        self.device = device
        self.tree_vertices = tree_vertices.to(device)

    def batch_transform(self, packings: torch.Tensor) -> torch.Tensor:
        """
        Transform tree vertices for batch of packings.

        Args:
            packings: (B, N, 3) tensor of (x, y, angle_rad)

        Returns:
            vertices: (B, N, 15, 2) transformed vertices
        """
        B, N, _ = packings.shape

        x = packings[:, :, 0]  # (B, N)
        y = packings[:, :, 1]  # (B, N)
        angles = packings[:, :, 2]  # (B, N)

        cos_a = torch.cos(angles)  # (B, N)
        sin_a = torch.sin(angles)  # (B, N)

        # Rotation matrices (B, N, 2, 2)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a], dim=2),
            torch.stack([sin_a, cos_a], dim=2)
        ], dim=2)

        # Rotate vertices: (B, N, 15, 2)
        # tree_vertices: (15, 2) -> (1, 1, 15, 2)
        v = self.tree_vertices.unsqueeze(0).unsqueeze(0)
        rotated = torch.einsum('bnij,mnkj->bnki', rot, v.expand(B, N, -1, -1))

        # Translate
        positions = torch.stack([x, y], dim=2).unsqueeze(2)  # (B, N, 1, 2)
        transformed = rotated + positions

        return transformed

    def batch_compute_sides(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute bounding box side for each packing.

        Args:
            vertices: (B, N, 15, 2)

        Returns:
            sides: (B,)
        """
        # Flatten to (B, N*15, 2)
        B = vertices.shape[0]
        flat = vertices.view(B, -1, 2)

        max_xy = flat.max(dim=1).values  # (B, 2)
        min_xy = flat.min(dim=1).values  # (B, 2)

        sizes = max_xy - min_xy  # (B, 2)
        sides = sizes.max(dim=1).values  # (B,)

        return sides

    def batch_check_overlaps(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Check for overlaps in each packing (approximation).

        Args:
            vertices: (B, N, 15, 2)

        Returns:
            has_overlap: (B,) boolean tensor
        """
        B, N, _, _ = vertices.shape

        # Compute centroids
        centroids = vertices.mean(dim=2)  # (B, N, 2)

        # Pairwise distances
        # (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2) -> (B, N, N)
        diff = centroids.unsqueeze(2) - centroids.unsqueeze(1)
        dists = torch.norm(diff, dim=3)  # (B, N, N)

        # Approximate collision threshold
        min_dist = 0.5  # Trees closer than this likely overlap

        # Check upper triangle only
        mask = torch.triu(torch.ones(N, N, device=self.device), diagonal=1).bool()
        masked_dists = dists[:, mask]  # (B, N*(N-1)/2)

        has_overlap = (masked_dists < min_dist).any(dim=1)  # (B,)

        return has_overlap

    def evaluate_batch(self, packings: torch.Tensor) -> torch.Tensor:
        """
        Evaluate batch of packings.

        Args:
            packings: (B, N, 3) tensor

        Returns:
            scores: (B,) tensor (inf if overlap)
        """
        vertices = self.batch_transform(packings)
        sides = self.batch_compute_sides(vertices)
        has_overlap = self.batch_check_overlaps(vertices)

        # Penalize overlaps
        sides[has_overlap] = float('inf')

        return sides

    def random_search(self, n: int, num_samples: int = 100000,
                      batch_size: int = 10000) -> tuple:
        """
        Random search with GPU acceleration.
        """
        best_score = float('inf')
        best_packing = None

        for batch_start in range(0, num_samples, batch_size):
            # Generate random packings
            packings = torch.randn(batch_size, n, 3, device=self.device)
            packings[:, :, :2] *= 3.0  # Scale positions
            packings[:, :, 2] *= 2 * np.pi  # Random angles

            # Evaluate
            scores = self.evaluate_batch(packings)

            # Update best
            min_idx = scores.argmin()
            if scores[min_idx] < best_score:
                best_score = scores[min_idx].item()
                best_packing = packings[min_idx].cpu().numpy()

        return best_score, best_packing
```

### Success Criteria
- [ ] Evaluate 100K packings in <10 seconds
- [ ] Random search finds feasible solutions
- [ ] Supports Apple Silicon (MPS) and CUDA

### Deliverables
- `python/gpu_batch_eval.py` - GPU evaluation module
- Benchmark: throughput (packings/second)

---

## Gen115: MCTS for Sequential Placement

### Objective
Use Monte Carlo Tree Search to plan tree placements with lookahead.

### Implementation

```python
# python/mcts_packing.py
import numpy as np
from typing import List, Tuple, Optional
import math

class MCTSNode:
    def __init__(self, state: List[dict], parent=None, action=None):
        self.state = state  # List of placed trees
        self.parent = parent
        self.action = action  # (x, y, angle) that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_terminal(self, n: int) -> bool:
        return len(self.state) >= n

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return -self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb1(c))

    def best_action_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.visits)

class PackingMCTS:
    def __init__(self, n: int, action_samples: int = 50):
        self.n = n
        self.action_samples = action_samples

    def get_actions(self, state: List[dict]) -> List[Tuple[float, float, float]]:
        """Generate candidate placements for next tree."""
        if not state:
            # First tree: place at origin with different angles
            return [(0, 0, a * 45) for a in range(8)]

        # Sample placements around existing trees
        actions = []
        bounds = compute_bounds(state)

        for _ in range(self.action_samples):
            # Random position near boundary
            x = np.random.uniform(bounds['min_x'] - 1, bounds['max_x'] + 1)
            y = np.random.uniform(bounds['min_y'] - 1, bounds['max_y'] + 1)
            angle = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])

            # Check if valid
            candidate = state + [{'x': x, 'y': y, 'angle': angle}]
            if not has_any_overlap(candidate):
                actions.append((x, y, angle))

        return actions if actions else [(0, 0, 0)]  # Fallback

    def rollout(self, state: List[dict]) -> float:
        """Random rollout to terminal state."""
        current = list(state)

        while len(current) < self.n:
            actions = self.get_actions(current)
            if not actions:
                return float('inf')  # Failed rollout

            action = actions[np.random.randint(len(actions))]
            current.append({'x': action[0], 'y': action[1], 'angle': action[2]})

        return -compute_side_length(current)  # Negative because we maximize

    def search(self, initial_state: List[dict] = None,
               simulations: int = 1000) -> List[dict]:
        """Run MCTS to find best packing."""
        root = MCTSNode(initial_state or [])

        for _ in range(simulations):
            node = root

            # Selection
            while not node.is_terminal(self.n) and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.is_terminal(self.n):
                if node.untried_actions is None:
                    node.untried_actions = self.get_actions(node.state)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    new_state = node.state + [{'x': action[0], 'y': action[1], 'angle': action[2]}]
                    child = MCTSNode(new_state, parent=node, action=action)
                    node.children.append(child)
                    node = child

            # Simulation
            value = self.rollout(node.state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # Extract best path
        result = []
        node = root
        while node.children:
            node = node.best_action_child()
            if node.action:
                result.append({'x': node.action[0], 'y': node.action[1], 'angle': node.action[2]})

        return result

def mcts_pack(n: int, simulations: int = 5000) -> Tuple[float, List[dict]]:
    """Pack n trees using MCTS."""
    mcts = PackingMCTS(n)
    trees = mcts.search(simulations=simulations)

    if len(trees) < n:
        # MCTS didn't complete, fill with greedy
        trees = greedy_complete(trees, n)

    return compute_side_length(trees), trees
```

### Success Criteria
- [ ] MCTS completes for n=10, 20 within reasonable time
- [ ] Beats random greedy on at least 30% of cases
- [ ] Scales to n=50 with reduced simulations

### Deliverables
- `python/mcts_packing.py` - MCTS implementation

---

## Gen116: Genetic Algorithm with Smart Crossover

### Objective
Evolve packings using crossover that preserves good spatial clusters.

### Implementation

```python
# python/genetic_packing.py
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

class PackingGA:
    def __init__(self, n: int, pop_size: int = 100,
                 mutation_rate: float = 0.1, elite_size: int = 5):
        self.n = n
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def initialize_population(self) -> List[List[dict]]:
        """Create initial population with diverse strategies."""
        population = []

        for _ in range(self.pop_size):
            # Use different initialization strategies
            strategy = np.random.choice(['greedy', 'random', 'spiral', 'grid'])

            if strategy == 'greedy':
                trees = greedy_pack(self.n)
            elif strategy == 'random':
                trees = random_pack(self.n)
            elif strategy == 'spiral':
                trees = spiral_pack(self.n)
            else:
                trees = grid_pack(self.n)

            population.append(trees)

        return population

    def fitness(self, trees: List[dict]) -> float:
        """Fitness = negative side length (maximize fitness = minimize side)."""
        if has_any_overlap(trees):
            return -1000.0  # Heavy penalty
        return -compute_side_length(trees)

    def cluster_trees(self, trees: List[dict], k: int) -> List[List[int]]:
        """Cluster trees spatially using K-means."""
        if len(trees) < k:
            return [[i] for i in range(len(trees))]

        positions = np.array([[t['x'], t['y']] for t in trees])
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(positions)

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        return clusters

    def crossover(self, parent1: List[dict], parent2: List[dict]) -> List[dict]:
        """
        Cluster-based crossover.

        1. Cluster each parent into k groups
        2. Select clusters from each parent
        3. Repair overlaps
        """
        k = max(2, self.n // 10)

        clusters1 = self.cluster_trees(parent1, k)
        clusters2 = self.cluster_trees(parent2, k)

        # Build child by selecting clusters
        child_indices = []
        used_from_p1 = set()
        used_from_p2 = set()

        for i in range(k):
            if np.random.random() < 0.5:
                # Take cluster from parent1
                for idx in clusters1[i]:
                    if idx not in used_from_p1:
                        child_indices.append(('p1', idx))
                        used_from_p1.add(idx)
            else:
                # Take cluster from parent2
                for idx in clusters2[i]:
                    if idx not in used_from_p2:
                        child_indices.append(('p2', idx))
                        used_from_p2.add(idx)

        # Build child trees
        child = []
        for source, idx in child_indices[:self.n]:
            if source == 'p1':
                child.append(parent1[idx].copy())
            else:
                child.append(parent2[idx].copy())

        # Fill remaining with greedy
        while len(child) < self.n:
            child.append(greedy_place_one(child))

        # Repair overlaps
        return self.repair(child[:self.n])

    def mutate(self, trees: List[dict]) -> List[dict]:
        """Apply mutation operators."""
        trees = [t.copy() for t in trees]

        mutation_type = np.random.choice(['translate', 'rotate', 'swap', 'sa_burst'])

        if mutation_type == 'translate':
            # Move random tree
            idx = np.random.randint(len(trees))
            trees[idx]['x'] += np.random.randn() * 0.1
            trees[idx]['y'] += np.random.randn() * 0.1

        elif mutation_type == 'rotate':
            # Rotate random tree
            idx = np.random.randint(len(trees))
            trees[idx]['angle'] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])

        elif mutation_type == 'swap':
            # Swap two trees
            i, j = np.random.choice(len(trees), 2, replace=False)
            trees[i]['x'], trees[j]['x'] = trees[j]['x'], trees[i]['x']
            trees[i]['y'], trees[j]['y'] = trees[j]['y'], trees[i]['y']

        else:
            # SA burst: run short SA
            trees = sa_refine(trees, iterations=500)

        return self.repair(trees)

    def repair(self, trees: List[dict]) -> List[dict]:
        """Repair overlapping configuration."""
        for _ in range(100):
            if not has_any_overlap(trees):
                break

            for i in range(len(trees)):
                for j in range(i + 1, len(trees)):
                    if polygons_overlap(trees[i], trees[j]):
                        # Push apart
                        dx = trees[j]['x'] - trees[i]['x']
                        dy = trees[j]['y'] - trees[i]['y']
                        dist = np.sqrt(dx*dx + dy*dy) + 0.001

                        trees[j]['x'] += dx / dist * 0.05
                        trees[j]['y'] += dy / dist * 0.05

        return trees

    def evolve(self, generations: int = 100) -> Tuple[float, List[dict]]:
        """Run genetic algorithm."""
        population = self.initialize_population()

        best_fitness = -float('inf')
        best_trees = None

        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(ind) for ind in population]

            # Update best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_trees = population[gen_best_idx]

            # Selection (tournament)
            selected = []
            for _ in range(self.pop_size - self.elite_size):
                tournament = np.random.choice(self.pop_size, 3, replace=False)
                winner = max(tournament, key=lambda i: fitness_scores[i])
                selected.append(population[winner])

            # Elitism
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]

            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1 = self.crossover(selected[i], selected[i+1])
                child2 = self.crossover(selected[i+1], selected[i])
                offspring.extend([child1, child2])

            # Mutation
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_rate:
                    offspring[i] = self.mutate(offspring[i])

            # New population
            population = elites + offspring[:self.pop_size - self.elite_size]

            if gen % 10 == 0:
                print(f"Gen {gen}: best fitness = {best_fitness:.4f}")

        return -best_fitness, best_trees
```

### Success Criteria
- [ ] GA maintains population diversity
- [ ] Crossover produces valid offspring >50% of time
- [ ] Beats single-run greedy on average

### Deliverables
- `python/genetic_packing.py` - GA implementation

---

## Gen117: Learned Value Function

### Objective
Train a neural network to predict final packing quality from partial state.

### Implementation

```python
# python/value_network.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PackingDataset(Dataset):
    """Dataset of (partial_packing, n_target, final_side_length) tuples."""

    def __init__(self, data_file: str):
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        partial, n_target, final_side = self.data[idx]
        return partial, n_target, final_side

class SetEncoder(nn.Module):
    """Permutation-invariant encoder for set of trees."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()

        self.tree_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, trees: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            trees: (B, max_trees, 3) - padded tree features
            mask: (B, max_trees) - True for valid trees

        Returns:
            encoding: (B, output_dim)
        """
        # Encode each tree
        encoded = self.tree_encoder(trees)  # (B, T, hidden)

        # Self-attention with masking
        if mask is not None:
            attn_mask = ~mask  # True = ignore
        else:
            attn_mask = None

        attended, _ = self.attention(encoded, encoded, encoded, key_padding_mask=attn_mask)

        # Pool (mean of valid trees)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = attended.mean(dim=1)

        return self.output_proj(pooled)

class PackingValueNetwork(nn.Module):
    """Predicts final side length from partial packing."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.set_encoder = SetEncoder(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),  # +1 for n_target
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, trees: torch.Tensor, mask: torch.Tensor, n_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trees: (B, max_trees, 3)
            mask: (B, max_trees)
            n_target: (B,) target number of trees

        Returns:
            predicted_side: (B,)
        """
        encoding = self.set_encoder(trees, mask)  # (B, hidden)

        # Concatenate n_target
        n_normalized = n_target.unsqueeze(1).float() / 200.0  # Normalize
        combined = torch.cat([encoding, n_normalized], dim=1)

        return self.value_head(combined).squeeze(-1)

def generate_training_data(num_samples: int = 100000) -> list:
    """Generate training data by running packings and recording trajectories."""
    data = []

    for _ in range(num_samples):
        n = np.random.randint(5, 201)

        # Run greedy packing
        trees = []
        trajectory = []

        for i in range(n):
            # Record partial state
            if len(trees) > 0:
                partial = torch.tensor([[t['x'], t['y'], t['angle']/360] for t in trees])
            else:
                partial = torch.zeros(0, 3)

            # Place next tree
            next_tree = greedy_place_one(trees)
            trees.append(next_tree)

            trajectory.append(partial)

        # Final side length
        final_side = compute_side_length(trees)

        # Add samples from trajectory
        for partial in trajectory[::5]:  # Sample every 5th state
            data.append((partial, n, final_side))

    return data

def train_value_network(data_file: str, epochs: int = 50, batch_size: int = 256):
    """Train value network on generated data."""
    dataset = PackingDataset(data_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PackingValueNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for partial, n_target, final_side in loader:
            # Pad partial packings
            max_len = max(p.shape[0] for p in partial)
            padded = torch.zeros(len(partial), max_len, 3)
            mask = torch.zeros(len(partial), max_len, dtype=torch.bool)

            for i, p in enumerate(partial):
                if p.shape[0] > 0:
                    padded[i, :p.shape[0]] = p
                    mask[i, :p.shape[0]] = True

            # Forward
            pred = model(padded, mask, n_target)
            loss = criterion(pred, final_side)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")

    return model
```

### Success Criteria
- [ ] Value network achieves <10% MAPE on test set
- [ ] Guides greedy to better solutions than random
- [ ] Generalizes across different n values

### Deliverables
- `python/value_network.py` - Network architecture and training
- `models/value_net.pt` - Trained model

---

## Gen118: Hybrid Pipeline Integration

### Objective
Combine all techniques into a unified pipeline that selects the best approach for each n.

### Implementation

```python
# python/hybrid_pipeline.py
from typing import List, Tuple, Dict
import time

class HybridPacker:
    def __init__(self):
        # Load all components
        self.nfp_table = load_nfp_table()
        self.value_net = load_value_network()
        self.gpu_evaluator = GPUPackingEvaluator()

    def pack(self, n: int, time_budget: float = 60.0) -> Tuple[float, List[dict]]:
        """
        Pack n trees using best available method.

        Strategy:
        - n <= 10: Try ILP first (may find optimal)
        - n <= 30: Use CMA-ES with longer budget
        - n <= 100: Use value-guided greedy + CMA-ES refinement
        - n > 100: Use fast greedy + SA refinement
        """
        start = time.time()
        candidates = []

        # Always try fast greedy as baseline
        greedy_trees = self.rust_greedy(n)
        candidates.append(('greedy', greedy_trees))

        # ILP for small n
        if n <= 10:
            ilp_result = self.ilp_optimal(n, timeout=min(30, time_budget * 0.5))
            if ilp_result:
                candidates.append(('ilp', ilp_result))

        # NFP-based greedy
        if n <= 50:
            nfp_trees = self.nfp_greedy(n)
            candidates.append(('nfp', nfp_trees))

        # CMA-ES refinement of best so far
        if time.time() - start < time_budget * 0.7:
            best_so_far = min(candidates, key=lambda x: compute_side_length(x[1]))[1]
            remaining = time_budget - (time.time() - start)

            cmaes_trees = self.cmaes_refine(best_so_far, time_budget=remaining * 0.8)
            candidates.append(('cmaes', cmaes_trees))

        # GPU parallel search (if time permits)
        if time.time() - start < time_budget * 0.9 and n <= 30:
            gpu_trees = self.gpu_random_search(n, samples=50000)
            if gpu_trees:
                candidates.append(('gpu', gpu_trees))

        # Select best
        best_name, best_trees = min(candidates, key=lambda x: compute_side_length(x[1]))
        best_side = compute_side_length(best_trees)

        print(f"n={n}: best={best_name} side={best_side:.4f}")

        return best_side, best_trees

    def pack_all(self, max_n: int = 200, time_per_n: float = 30.0) -> Dict[int, List[dict]]:
        """Pack all n=1..max_n."""
        results = {}
        total_score = 0.0

        for n in range(1, max_n + 1):
            # Allocate more time to small n (higher score impact)
            time_budget = time_per_n * (1 + 10 / n)

            side, trees = self.pack(n, time_budget=time_budget)
            results[n] = trees
            total_score += side ** 2 / n

            if n % 20 == 0:
                print(f"Progress: n={n}, running score={total_score:.4f}")

        print(f"Final score: {total_score:.4f}")
        return results
```

### Success Criteria
- [ ] Pipeline beats any single method on average
- [ ] Automatic method selection works correctly
- [ ] Total score < 80

### Deliverables
- `python/hybrid_pipeline.py` - Integrated pipeline
- Full submission with method breakdown

---

## Gen119: Final Optimization Push

### Objective
Final refinements and ensemble of best solutions.

### Tasks

1. **Solution Ensemble**: Run each method multiple times, keep best per n
2. **Parameter Tuning**: Grid search over CMA-ES, GA parameters
3. **Long ILP Runs**: Overnight ILP for n=1..20
4. **Post-processing**: Apply all refinement techniques to best solutions

### Implementation

```python
# python/final_ensemble.py
def ensemble_optimize(n: int, methods: list, runs_per_method: int = 5) -> Tuple[float, List[dict]]:
    """Run multiple methods multiple times, keep best."""
    all_results = []

    for method_name, method_fn in methods:
        for run in range(runs_per_method):
            side, trees = method_fn(n)
            all_results.append((method_name, run, side, trees))

    # Select best
    best = min(all_results, key=lambda x: x[2])
    return best[2], best[3]

def generate_final_submission():
    """Generate final competition submission."""
    methods = [
        ('rust_greedy', rust_greedy_pack),
        ('nfp_greedy', nfp_greedy_pack),
        ('cmaes', cmaes_pack),
        ('mcts', mcts_pack),
        ('ga', ga_pack),
    ]

    results = {}

    for n in range(1, 201):
        # More runs for small n
        runs = 10 if n <= 20 else 5 if n <= 50 else 3

        side, trees = ensemble_optimize(n, methods, runs_per_method=runs)
        results[n] = trees

        print(f"n={n}: side={side:.4f}")

    # Write submission
    write_submission(results, 'submission_final.csv')

    # Compute score
    score = sum(compute_side_length(results[n])**2 / n for n in range(1, 201))
    print(f"Final score: {score:.4f}")
```

### Success Criteria
- [ ] Score < 78 (10% improvement from 86.13)
- [ ] Stretch: Score < 75 (13% improvement)
- [ ] Ultimate: Score < 72 (within 5% of leader)

### Deliverables
- `submission_final.csv` - Best submission
- Method breakdown showing contribution of each technique

---

## Summary Timeline

| Gen | Focus | Target Score | Key Technique |
|-----|-------|--------------|---------------|
| 110 | NFP Foundation | 85 | No-Fit Polygon |
| 111 | CMA-ES | 84 | Global optimization |
| 112 | ILP Small n | 83 | Exact solutions n≤15 |
| 113 | Differentiable | 82.5 | Gradient descent |
| 114 | GPU Batch | 82 | Parallel evaluation |
| 115 | MCTS | 81.5 | Sequential planning |
| 116 | Genetic | 81 | Evolutionary crossover |
| 117 | Value Network | 80 | Learned heuristics |
| 118 | Hybrid | 79 | Integrated pipeline |
| 119 | Final Push | <78 | Ensemble + tuning |

---

## Dependencies

```bash
pip install cma shapely ortools torch scikit-learn numpy
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NFP too slow | Medium | High | Precompute, cache |
| CMA-ES stuck infeasible | Low | Medium | Repair operator |
| ILP timeout | High (large n) | Low | Use only for small n |
| GPU memory limits | Medium | Low | Batch processing |
| Value network overfits | Medium | Medium | More training data |

---

## Notes for Implementation

1. **Start with Gen110 (NFP)** - it's foundational for ILP and other methods
2. **Each generation builds on previous** - don't skip ahead
3. **Keep Rust baseline** - always compare to rust_greedy
4. **Track all results** - maintain score history for each n
5. **Time-box each generation** - 2-3 hours max, move on if stuck
