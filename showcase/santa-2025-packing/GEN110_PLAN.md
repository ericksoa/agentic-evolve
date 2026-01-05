# Gen110 Plan: Frontier Optimization Strategies

## Current Status
- **Best Score**: 86.13 (Gen103)
- **Leader (#1)**: ~69
- **Gap**: 25% - significant room for improvement
- **Key Insight**: We're stuck in a local optimum. Need fundamentally different approaches.

## Analysis: Why Current Approach Plateaued

Our current approach (greedy placement + SA refinement) has fundamental limitations:
1. **Sequential placement bias**: Each tree placement constrains future options
2. **Local search only**: SA explores neighborhood but can't escape deep local optima
3. **No global structure learning**: Doesn't learn patterns across different n values
4. **Single-objective**: Only optimizes side length, ignores packing density structure

## Frontier Strategies

### Strategy A: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
**Potential Impact: HIGH (5-15%)**

CMA-ES is state-of-the-art for continuous black-box optimization. Treat packing as optimizing 3n parameters (x, y, θ for each tree).

```python
import cma

def pack_n_trees_cmaes(n: int, max_evals: int = 10000) -> Tuple[float, List]:
    """
    Use CMA-ES to optimize tree positions globally.

    Key advantages:
    - Learns covariance structure (which moves work together)
    - Handles non-separable objectives well
    - Self-adapting step sizes
    """
    dim = 3 * n  # x, y, angle for each tree

    def objective(params):
        trees = params_to_trees(params, n)
        if has_any_overlap(trees):
            return 1000.0 + overlap_penalty(trees)  # Soft constraint
        return compute_side_length(trees)

    # Initialize from greedy solution
    initial = trees_to_params(greedy_pack(n))

    es = cma.CMAEvolutionStrategy(initial, 0.3, {
        'maxfevals': max_evals,
        'popsize': 4 + int(3 * np.log(dim)),
        'bounds': [[-10]*dim, [10]*dim],
    })

    es.optimize(objective)
    return es.result.fbest, params_to_trees(es.result.xbest)
```

**Why it might work**: CMA-ES learns which tree movements are correlated, enabling coordinated moves that SA can't discover.

---

### Strategy B: Differentiable Packing with Gradient Descent
**Potential Impact: MEDIUM-HIGH (3-10%)**

Make the packing problem differentiable using soft overlap penalties and optimize with Adam/L-BFGS.

```python
import torch

class DifferentiablePacking(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        # Learnable parameters
        self.positions = torch.nn.Parameter(torch.randn(n, 2) * 0.5)
        self.angles = torch.nn.Parameter(torch.rand(n) * 360)

    def forward(self):
        # Compute soft bounding box
        vertices = self.get_all_vertices()  # (n, 15, 2)
        min_xy = vertices.min(dim=0).values.min(dim=0).values
        max_xy = vertices.max(dim=0).values.max(dim=0).values
        side = torch.max(max_xy - min_xy)

        # Soft overlap penalty (differentiable approximation)
        overlap_loss = self.compute_soft_overlap()

        return side + 100.0 * overlap_loss

    def compute_soft_overlap(self):
        """Differentiable overlap using signed distance or GJK approximation"""
        # Use polygon SDF or learned overlap network
        pass

# Optimize
model = DifferentiablePacking(n=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(5000):
    loss = model()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Project to feasible (snap to valid if needed)
    with torch.no_grad():
        model.snap_to_valid()
```

**Why it might work**: Gradient information enables much faster convergence than random search.

---

### Strategy C: Monte Carlo Tree Search (MCTS) for Sequential Placement
**Potential Impact: MEDIUM-HIGH (3-8%)**

Treat tree placement as a sequential decision problem. Use MCTS with learned value/policy networks.

```python
class PackingMCTS:
    """
    MCTS for tree packing.

    State: Current partial packing (list of placed trees)
    Action: (x, y, angle) for next tree
    Reward: -side_length at terminal state
    """

    def __init__(self, n: int, policy_net=None, value_net=None):
        self.n = n
        self.policy_net = policy_net  # Suggests promising placements
        self.value_net = value_net    # Estimates final packing quality

    def search(self, state: List[PlacedTree], simulations: int = 1000):
        root = MCTSNode(state)

        for _ in range(simulations):
            node = root

            # Selection (UCB1)
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(c=1.414)

            # Expansion
            if not node.is_terminal():
                action = self.select_action(node)
                node = node.expand(action)

            # Simulation (rollout with policy net or random)
            value = self.rollout(node.state)

            # Backpropagation
            node.backpropagate(value)

        return root.best_action()

    def select_action(self, node):
        """Use policy network to suggest actions, or discretized grid"""
        if self.policy_net:
            return self.policy_net.suggest(node.state)
        else:
            # Grid-based action space
            return self.sample_valid_placements(node.state, k=10)
```

**Why it might work**: Looks ahead to evaluate placement quality, avoiding greedy mistakes.

---

### Strategy D: NFP (No-Fit Polygon) Based Exact Placement
**Potential Impact: HIGH (5-15%)**

NFP is the gold standard in cutting & packing optimization. For each pair of polygons, compute the region where one cannot be placed relative to the other.

```python
from shapely.geometry import Polygon
from shapely.ops import unary_union

def compute_nfp(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """
    Compute No-Fit Polygon: the locus of positions where poly_b
    touches but doesn't overlap poly_a.

    Uses Minkowski sum: NFP = A ⊕ (-B)
    """
    # Reflect B around origin
    reflected_b = scale(poly_b, -1, -1, origin=(0, 0))

    # Minkowski sum
    nfp = minkowski_sum(poly_a, reflected_b)
    return nfp

def nfp_based_placement(existing: List[PlacedTree], new_angle: float) -> List[Tuple[float, float]]:
    """
    Find ALL valid placement positions for a new tree using NFP.

    Returns the boundary of the feasible region.
    """
    new_tree_poly = get_tree_polygon(0, 0, new_angle)

    # Compute NFP with each existing tree
    forbidden_regions = []
    for tree in existing:
        existing_poly = get_tree_polygon(tree.x, tree.y, tree.angle)
        nfp = compute_nfp(existing_poly, new_tree_poly)
        forbidden_regions.append(nfp)

    # Union of all forbidden regions
    total_forbidden = unary_union(forbidden_regions)

    # Valid positions are on the boundary of forbidden region
    # (touching but not overlapping)
    return get_boundary_positions(total_forbidden)
```

**Why it might work**: NFP gives exact characterization of valid placements, enabling provably optimal local decisions.

---

### Strategy E: Genetic Algorithm with Crossover on Packings
**Potential Impact: MEDIUM (2-5%)**

Evolve a population of packings with crossover operators that preserve good sub-structures.

```python
class PackingGA:
    def __init__(self, n: int, pop_size: int = 100):
        self.n = n
        self.pop_size = pop_size
        self.population = [self.random_packing() for _ in range(pop_size)]

    def crossover(self, parent1: List[PlacedTree], parent2: List[PlacedTree]) -> List[PlacedTree]:
        """
        Intelligent crossover preserving spatial clusters.

        1. Identify clusters in each parent (trees that are close together)
        2. Transfer whole clusters between parents
        3. Repair overlaps with local search
        """
        # K-means clustering on tree positions
        clusters1 = cluster_trees(parent1, k=self.n // 5)
        clusters2 = cluster_trees(parent2, k=self.n // 5)

        # Randomly select clusters from each parent
        child_clusters = []
        for i in range(len(clusters1)):
            if random.random() < 0.5:
                child_clusters.extend(clusters1[i])
            else:
                child_clusters.extend(clusters2[i])

        # Repair: resolve overlaps with local search
        return repair_packing(child_clusters)

    def mutate(self, packing: List[PlacedTree]) -> List[PlacedTree]:
        """
        Mutations:
        1. Swap two trees
        2. Rotate subset
        3. Translate subset
        4. SA refinement burst
        """
        mutation_type = random.choice(['swap', 'rotate_subset', 'translate', 'sa_burst'])
        # ... implement each
```

**Why it might work**: Crossover can combine good regions from different packings.

---

### Strategy F: GPU-Accelerated Massive Parallel Search
**Potential Impact: MEDIUM-HIGH (3-8%)**

Use GPU to evaluate thousands of candidate packings simultaneously.

```python
import torch

class GPUPackingEvaluator:
    def __init__(self, n: int, batch_size: int = 10000):
        self.n = n
        self.batch_size = batch_size
        self.tree_vertices = torch.tensor(TREE_VERTICES, device='cuda')  # (15, 2)

    def evaluate_batch(self, packings: torch.Tensor) -> torch.Tensor:
        """
        Evaluate batch_size packings in parallel on GPU.

        Args:
            packings: (batch_size, n, 3) tensor of (x, y, angle)

        Returns:
            scores: (batch_size,) tensor of side lengths (inf if overlap)
        """
        # Transform all trees for all packings
        # packings: (B, N, 3) -> vertices: (B, N, 15, 2)
        transformed = self.batch_transform(packings)

        # Compute bounding boxes
        mins = transformed.min(dim=2).values.min(dim=1).values  # (B, 2)
        maxs = transformed.max(dim=2).values.max(dim=1).values  # (B, 2)
        sides = (maxs - mins).max(dim=1).values  # (B,)

        # Check overlaps (parallel SAT or GJK)
        has_overlap = self.batch_check_overlaps(transformed)  # (B,)

        # Penalize overlaps
        sides[has_overlap] = float('inf')

        return sides

    def random_search(self, num_samples: int = 1_000_000) -> Tuple[float, List]:
        """Generate and evaluate 1M random packings on GPU"""
        best_score = float('inf')
        best_packing = None

        for batch_start in range(0, num_samples, self.batch_size):
            # Generate random packings
            packings = self.generate_random_batch()

            # Evaluate
            scores = self.evaluate_batch(packings)

            # Update best
            min_idx = scores.argmin()
            if scores[min_idx] < best_score:
                best_score = scores[min_idx].item()
                best_packing = packings[min_idx].cpu().numpy()

        return best_score, best_packing
```

**Why it might work**: Brute force with massive parallelism can find surprisingly good solutions.

---

### Strategy G: Learn a Packing Value Function
**Potential Impact: HIGH (5-15%)**

Train a neural network to predict final packing quality from partial state, then use it to guide search.

```python
class PackingValueNetwork(torch.nn.Module):
    """
    Predicts final side length given current partial packing.

    Input: Set of placed trees (variable size)
    Output: Predicted optimal final side length
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Encode each tree
        self.tree_encoder = nn.Sequential(
            nn.Linear(3, 64),  # x, y, angle
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        # Set aggregation (permutation invariant)
        self.attention = nn.MultiheadAttention(128, 4)

        # Predict value
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, trees: torch.Tensor, n_target: int) -> torch.Tensor:
        """
        Args:
            trees: (batch, num_placed, 3) - placed trees
            n_target: target number of trees

        Returns:
            value: (batch,) - predicted final side length
        """
        # Encode trees
        encoded = self.tree_encoder(trees)  # (B, K, 128)

        # Self-attention aggregation
        aggregated, _ = self.attention(encoded, encoded, encoded)
        pooled = aggregated.mean(dim=1)  # (B, 128)

        # Predict
        return self.value_head(pooled).squeeze(-1)

# Training: generate packings, record trajectories, train to predict final score
```

**Why it might work**: Learned heuristics can capture complex patterns that hand-crafted heuristics miss.

---

### Strategy H: Hybrid ILP + Heuristic
**Potential Impact: VERY HIGH for small n (10-20% for n≤20)**

Use Integer Linear Programming for small n where it can find provably optimal solutions.

```python
from ortools.sat.python import cp_model

def ilp_optimal_packing(n: int, grid_res: int = 100, angle_steps: int = 8, timeout: int = 300):
    """
    Find provably optimal packing using ILP.

    Discretization:
    - Position: grid_res x grid_res grid
    - Rotation: angle_steps angles (0, 45, 90, ...)

    Variables:
    - x[i], y[i]: position of tree i (integer grid coordinates)
    - r[i]: rotation index of tree i
    - s: side length (minimize)

    Constraints:
    - All trees within [0, s] x [0, s]
    - No overlaps (via precomputed NFP lookup tables)
    """
    model = cp_model.CpModel()

    # Precompute NFP for all rotation pairs
    nfp_table = precompute_nfp_table(angle_steps, grid_res)

    # Variables
    x = [model.NewIntVar(0, grid_res, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, grid_res, f'y_{i}') for i in range(n)]
    r = [model.NewIntVar(0, angle_steps - 1, f'r_{i}') for i in range(n)]
    s = model.NewIntVar(1, grid_res, 'side')

    # Containment constraints
    for i in range(n):
        for ri in range(angle_steps):
            bounds = tree_bounds_at_rotation(ri, grid_res)
            # If r[i] == ri, then x[i] + bounds.max_x <= s, etc.
            # Use indicator constraints
            ...

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # For each rotation pair, add NFP constraint
            for ri in range(angle_steps):
                for rj in range(angle_steps):
                    add_nfp_constraint(model, x[i], y[i], x[j], y[j],
                                       r[i], ri, r[j], rj, nfp_table[ri][rj])

    model.Minimize(s)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return extract_solution(solver, x, y, r, grid_res, angle_steps)
    return None
```

**Why it might work**: ILP can find globally optimal solutions for small n, eliminating score loss from suboptimal small packings.

---

## Implementation Priority

| Strategy | Effort | Potential Gain | Risk | Priority |
|----------|--------|----------------|------|----------|
| D: NFP-based | Medium | 5-15% | Low | **1** |
| A: CMA-ES | Low | 5-15% | Medium | **2** |
| H: ILP for small n | High | 10-20% (small n) | Low | **3** |
| B: Differentiable | Medium | 3-10% | Medium | **4** |
| F: GPU parallel | Medium | 3-8% | Low | **5** |
| G: Value network | High | 5-15% | High | 6 |
| C: MCTS | High | 3-8% | Medium | 7 |
| E: Genetic crossover | Medium | 2-5% | Medium | 8 |

---

## Recommended Gen110 Approach

### Phase 1: NFP Foundation (Priority 1)
Build proper NFP infrastructure - this is the mathematical foundation used by all serious packing solvers.

```
1. Implement Minkowski sum for tree polygon
2. Precompute NFP for all 8x8 rotation pairs
3. Use NFP to find ALL valid placements (not just sampled)
4. Select placement that minimizes side length increase
```

### Phase 2: CMA-ES Global Optimization (Priority 2)
Add CMA-ES as post-processing to escape local optima.

```
1. Install pycma library
2. Initialize from best greedy solution
3. Run CMA-ES with overlap penalty
4. Snap to valid solution with local search
```

### Phase 3: ILP for Small n (Priority 3)
Get provably optimal solutions for n=1..15.

```
1. Precompute discretized NFP tables
2. Formulate CP-SAT model
3. Solve with 5-minute timeout per n
4. Replace any Rust solutions that ILP beats
```

### Phase 4: Integration
Combine all approaches:
```
For n = 1..200:
    if n <= 15:
        Try ILP (5 min timeout)

    # Generate multiple candidates
    candidates = []
    candidates.append(rust_greedy(n))
    candidates.append(nfp_greedy(n))
    candidates.append(cmaes_optimize(best_candidate))

    # Select best
    best = min(candidates, key=side_length)
```

---

## Success Metrics

- [ ] NFP placement finds better solutions than Rust greedy for >50% of n
- [ ] CMA-ES improves at least 10% of n values post-greedy
- [ ] ILP finds provably optimal for n=1..10
- [ ] Combined score < 80 (7.5% improvement from 86.13)
- [ ] Stretch goal: score < 75 (13% improvement)

---

## Technical Requirements

```bash
# Python dependencies
pip install cma            # CMA-ES optimizer
pip install shapely        # Polygon operations / NFP
pip install ortools        # ILP solver
pip install torch          # GPU acceleration
pip install scipy          # Optimization utilities

# Rust dependencies (Cargo.toml)
rayon = "1.8"              # Parallel iteration
nalgebra = "0.32"          # Linear algebra
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| NFP computation too slow | Precompute and cache all rotation pairs |
| CMA-ES stuck in infeasible | Use repair operator + soft constraints |
| ILP timeout without solution | Use best feasible as fallback |
| GPU memory limits | Batch processing with streaming |

---

## Timeline Estimate

- Phase 1 (NFP): 3-4 hours
- Phase 2 (CMA-ES): 2-3 hours
- Phase 3 (ILP): 4-6 hours
- Phase 4 (Integration): 2-3 hours
- Full submission generation: 2-4 hours

Total: ~15-20 hours

---

## References

1. Burke et al. "A New Placement Heuristic for the Orthogonal Stock-Cutting Problem" (NFP)
2. Hansen & Ostermeier "Completely Derandomized Self-Adaptation in Evolution Strategies" (CMA-ES)
3. Bennell & Oliveira "The geometry of nesting problems" (NFP theory)
4. AlphaFold approach to structure prediction (learned value functions)
