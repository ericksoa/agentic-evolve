"""
lp_generator.py - Synthetic LP corpus generators.

Generates realistic LP instances for:
- Capacity planning problems
- Transportation/routing problems
- Supply chain optimization
- Resource allocation

All generators are parametrized and reproducible with fixed seeds.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import json

from .transforms import LPInstance


def lpinstance_to_lpdata(lp: 'LPInstance') -> 'LPData':
    """Convert LPInstance to LPData format."""
    from .mps_loader import LPData
    from scipy.sparse import csr_matrix

    # Reconstruct CSR matrix
    A = csr_matrix(
        (lp.A_data, lp.A_indices, lp.A_indptr),
        shape=(lp.n_cons, lp.n_vars)
    )

    return LPData(
        c=lp.c,
        A=A,
        b=lp.b,
        sense=lp.sense,
        lb=lp.lb,
        ub=lp.ub,
        name=lp.name,
    )


class LPType(Enum):
    """Types of LP problems we can generate."""
    TRANSPORTATION = "transportation"
    CAPACITY_PLANNING = "capacity_planning"
    RESOURCE_ALLOCATION = "resource_allocation"
    NETWORK_FLOW = "network_flow"
    BLENDING = "blending"
    PRODUCTION_PLANNING = "production_planning"


@dataclass
class GeneratorConfig:
    """Configuration for LP generation."""
    # Size parameters
    min_vars: int = 10
    max_vars: int = 1000
    min_cons: int = 5
    max_cons: int = 500

    # Density parameters
    min_density: float = 0.01
    max_density: float = 0.3

    # Coefficient parameters
    coef_min: float = 0.1
    coef_max: float = 100.0
    coef_log_scale: bool = True  # Use log-uniform distribution

    # RHS parameters
    rhs_scale: float = 100.0
    rhs_positive_only: bool = True

    # Bound parameters
    frac_bounded: float = 0.7
    frac_nonnegative: float = 0.9
    bound_max: float = 1000.0

    # Constraint sense distribution
    frac_leq: float = 0.6
    frac_eq: float = 0.2
    # frac_geq = 1 - frac_leq - frac_eq

    # Problem types to include
    problem_types: List[str] = None  # None = all types

    def __post_init__(self):
        if self.problem_types is None:
            self.problem_types = [t.value for t in LPType]


class LPGenerator:
    """
    Generates synthetic LP instances.

    Usage:
        gen = LPGenerator(seed=42, config=GeneratorConfig())
        lps = gen.generate_batch(100)
    """

    def __init__(self, seed: int = 42, config: Optional[GeneratorConfig] = None):
        self.rng = np.random.RandomState(seed)
        self.config = config or GeneratorConfig()
        self.instance_counter = 0

    def generate(self, lp_type: Optional[LPType] = None) -> LPInstance:
        """
        Generate a single LP instance.

        Args:
            lp_type: Specific type, or None for random

        Returns:
            LPInstance
        """
        if lp_type is None:
            type_name = self.rng.choice(self.config.problem_types)
            lp_type = LPType(type_name)

        self.instance_counter += 1

        if lp_type == LPType.TRANSPORTATION:
            return self._generate_transportation()
        elif lp_type == LPType.CAPACITY_PLANNING:
            return self._generate_capacity_planning()
        elif lp_type == LPType.RESOURCE_ALLOCATION:
            return self._generate_resource_allocation()
        elif lp_type == LPType.NETWORK_FLOW:
            return self._generate_network_flow()
        elif lp_type == LPType.BLENDING:
            return self._generate_blending()
        elif lp_type == LPType.PRODUCTION_PLANNING:
            return self._generate_production_planning()
        else:
            return self._generate_generic()

    def generate_batch(self, n: int, lp_type: Optional[LPType] = None) -> List[LPInstance]:
        """Generate a batch of LP instances."""
        return [self.generate(lp_type) for _ in range(n)]

    def generate_random_lp(
        self,
        n_vars_range: Tuple[int, int] = (10, 500),
        n_cons_range: Tuple[int, int] = (5, 300),
        density_range: Tuple[float, float] = (0.01, 0.3),
    ) -> 'LPData':
        """
        Generate a random LP as LPData (for evolution).

        Args:
            n_vars_range: (min_vars, max_vars)
            n_cons_range: (min_cons, max_cons)
            density_range: (min_density, max_density)

        Returns:
            LPData instance
        """
        # Temporarily override config
        old_min_vars, old_max_vars = self.config.min_vars, self.config.max_vars
        old_min_cons, old_max_cons = self.config.min_cons, self.config.max_cons
        old_min_density, old_max_density = self.config.min_density, self.config.max_density

        self.config.min_vars, self.config.max_vars = n_vars_range
        self.config.min_cons, self.config.max_cons = n_cons_range
        self.config.min_density, self.config.max_density = density_range

        try:
            lp_instance = self.generate()
            return lpinstance_to_lpdata(lp_instance)
        finally:
            # Restore config
            self.config.min_vars, self.config.max_vars = old_min_vars, old_max_vars
            self.config.min_cons, self.config.max_cons = old_min_cons, old_max_cons
            self.config.min_density, self.config.max_density = old_min_density, old_max_density

    def _random_size(self) -> Tuple[int, int]:
        """Generate random problem size."""
        n_vars = self.rng.randint(self.config.min_vars, self.config.max_vars + 1)
        n_cons = self.rng.randint(self.config.min_cons, self.config.max_cons + 1)
        return n_vars, n_cons

    def _random_density(self) -> float:
        """Generate random density."""
        return self.rng.uniform(self.config.min_density, self.config.max_density)

    def _random_coefficients(self, size: int) -> np.ndarray:
        """Generate random coefficients."""
        if self.config.coef_log_scale:
            log_min = np.log10(self.config.coef_min)
            log_max = np.log10(self.config.coef_max)
            log_vals = self.rng.uniform(log_min, log_max, size)
            vals = 10 ** log_vals
        else:
            vals = self.rng.uniform(self.config.coef_min, self.config.coef_max, size)

        # Random signs
        signs = self.rng.choice([-1, 1], size)
        return vals * signs

    def _random_sense(self, n_cons: int) -> np.ndarray:
        """Generate random constraint senses."""
        probs = [
            self.config.frac_leq,
            self.config.frac_eq,
            1 - self.config.frac_leq - self.config.frac_eq
        ]
        choices = self.rng.choice([-1, 0, 1], size=n_cons, p=probs)
        return choices.astype(np.int8)

    def _random_bounds(self, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random variable bounds."""
        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars, np.inf)

        # Non-negative variables
        nonneg_mask = self.rng.random(n_vars) < self.config.frac_nonnegative
        lb[nonneg_mask] = 0.0

        # Upper bounded variables
        bounded_mask = self.rng.random(n_vars) < self.config.frac_bounded
        ub[bounded_mask] = self.rng.uniform(1, self.config.bound_max, np.sum(bounded_mask))

        return lb, ub

    def _generate_transportation(self) -> LPInstance:
        """
        Generate a transportation problem.

        min sum_ij c_ij * x_ij
        s.t. sum_j x_ij <= supply_i  (supply constraints)
             sum_i x_ij >= demand_j  (demand constraints)
             x_ij >= 0
        """
        n_sources = self.rng.randint(5, 50)
        n_destinations = self.rng.randint(5, 50)
        n_vars = n_sources * n_destinations
        n_cons = n_sources + n_destinations

        # Cost coefficients (per unit shipping cost)
        c = self.rng.uniform(1, 100, n_vars)

        # Build constraint matrix
        # Supply constraints: sum_j x_ij <= supply_i
        # Demand constraints: sum_i x_ij >= demand_j

        A_data = []
        A_indices = []
        A_indptr = [0]

        # Supply constraints (one per source)
        for i in range(n_sources):
            for j in range(n_destinations):
                A_data.append(1.0)
                A_indices.append(i * n_destinations + j)
            A_indptr.append(len(A_data))

        # Demand constraints (one per destination)
        for j in range(n_destinations):
            for i in range(n_sources):
                A_data.append(1.0)
                A_indices.append(i * n_destinations + j)
            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS: supply and demand
        total_supply = self.rng.uniform(100, 10000)
        supplies = self.rng.dirichlet(np.ones(n_sources)) * total_supply * 1.1
        demands = self.rng.dirichlet(np.ones(n_destinations)) * total_supply

        b = np.concatenate([supplies, demands])

        # Sense: <= for supply, >= for demand
        sense = np.concatenate([
            np.full(n_sources, -1, dtype=np.int8),
            np.full(n_destinations, 1, dtype=np.int8)
        ])

        # Bounds: all variables non-negative
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"transportation_{self.instance_counter}",
        )

    def _generate_capacity_planning(self) -> LPInstance:
        """
        Generate a capacity planning problem.

        min sum_t (production_cost * x_t + holding_cost * inv_t)
        s.t. inv_{t-1} + x_t - inv_t = demand_t  (flow balance)
             x_t <= capacity_t  (capacity limits)
             x_t, inv_t >= 0
        """
        n_periods = self.rng.randint(10, 100)
        n_vars = 2 * n_periods  # production and inventory
        n_cons = 2 * n_periods  # balance and capacity

        # Costs
        prod_cost = self.rng.uniform(10, 100, n_periods)
        hold_cost = self.rng.uniform(1, 10, n_periods)
        c = np.concatenate([prod_cost, hold_cost])

        # Build constraints
        A_data = []
        A_indices = []
        A_indptr = [0]

        # Flow balance: inv_{t-1} + x_t - inv_t = demand_t
        for t in range(n_periods):
            # x_t coefficient: +1
            A_data.append(1.0)
            A_indices.append(t)

            # inv_t coefficient: -1
            A_data.append(-1.0)
            A_indices.append(n_periods + t)

            # inv_{t-1} coefficient: +1 (if t > 0)
            if t > 0:
                A_data.append(1.0)
                A_indices.append(n_periods + t - 1)

            A_indptr.append(len(A_data))

        # Capacity constraints: x_t <= capacity_t
        for t in range(n_periods):
            A_data.append(1.0)
            A_indices.append(t)
            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS
        demands = self.rng.uniform(10, 100, n_periods)
        capacities = self.rng.uniform(50, 200, n_periods)
        b = np.concatenate([demands, capacities])

        # Sense: = for balance, <= for capacity
        sense = np.concatenate([
            np.zeros(n_periods, dtype=np.int8),
            np.full(n_periods, -1, dtype=np.int8)
        ])

        # Bounds
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"capacity_{self.instance_counter}",
        )

    def _generate_resource_allocation(self) -> LPInstance:
        """
        Generate a resource allocation problem.

        max sum_j profit_j * x_j
        s.t. sum_j resource_ij * x_j <= available_i  (resource limits)
             x_j >= 0, x_j <= max_production_j
        """
        n_products = self.rng.randint(10, 100)
        n_resources = self.rng.randint(3, 20)
        n_vars = n_products
        n_cons = n_resources

        # Objective: maximize profit (negate for minimization)
        c = -self.rng.uniform(10, 1000, n_vars)

        # Resource usage matrix (sparse)
        density = self._random_density()
        nnz = max(1, int(n_vars * n_cons * density))

        # Generate sparse matrix
        A_data = []
        A_indices = []
        A_indptr = [0]

        for i in range(n_cons):
            # Each resource is used by some products
            n_products_using = max(1, int(n_vars * density * self.rng.uniform(0.5, 1.5)))
            products = self.rng.choice(n_vars, size=min(n_products_using, n_vars), replace=False)
            products.sort()

            for j in products:
                A_data.append(self.rng.uniform(0.1, 10))
                A_indices.append(j)

            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS: resource availability
        b = self.rng.uniform(100, 10000, n_cons)

        # Sense: all <= (resource limits)
        sense = np.full(n_cons, -1, dtype=np.int8)

        # Bounds
        lb = np.zeros(n_vars)
        ub = self.rng.uniform(10, 1000, n_vars)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"resource_{self.instance_counter}",
        )

    def _generate_network_flow(self) -> LPInstance:
        """
        Generate a minimum cost network flow problem.

        min sum_e cost_e * x_e
        s.t. sum_{e in delta+(v)} x_e - sum_{e in delta-(v)} x_e = supply_v
             0 <= x_e <= capacity_e

        Modified to ensure feasibility: we generate a feasible flow first,
        then derive capacities and supplies from it.
        """
        n_nodes = self.rng.randint(10, 50)
        # Edges: create a sparse connected graph
        n_edges = self.rng.randint(n_nodes, min(n_nodes * (n_nodes - 1) // 2, n_nodes * 3))

        n_vars = n_edges
        n_cons = n_nodes

        # Generate random edges (directed: i -> j)
        edges = []
        # First ensure connectivity with a spanning tree
        for i in range(1, n_nodes):
            j = self.rng.randint(0, i)
            # Direction from smaller to larger index
            edges.append((j, i))

        # Add random additional edges
        edge_set = set(edges)
        attempts = 0
        while len(edges) < n_edges and attempts < n_edges * 10:
            i = self.rng.randint(0, n_nodes)
            j = self.rng.randint(0, n_nodes)
            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                edges.append((i, j))
                edge_set.add((i, j))
            attempts += 1

        n_edges = len(edges)
        n_vars = n_edges

        # Cost coefficients
        c = self.rng.uniform(1, 100, n_vars)

        # Generate a feasible flow and derive supplies/capacities
        flow = self.rng.uniform(10, 50, n_vars)
        capacities = flow * self.rng.uniform(1.5, 3.0, n_vars)  # Capacity > flow

        # Compute net supply for each node from the flow
        supply = np.zeros(n_nodes)
        for e_idx, (i, j) in enumerate(edges):
            supply[i] -= flow[e_idx]  # outgoing
            supply[j] += flow[e_idx]  # incoming

        # Build incidence matrix
        A_data = []
        A_indices = []
        A_indptr = [0]

        for v in range(n_nodes):
            for e_idx, (i, j) in enumerate(edges):
                if i == v:
                    A_data.append(-1.0)  # outgoing (flow leaves)
                    A_indices.append(e_idx)
                elif j == v:
                    A_data.append(1.0)  # incoming (flow enters)
                    A_indices.append(e_idx)
            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # Supply is negative (source) or positive (sink)
        b = supply

        # Sense: all = (flow conservation)
        sense = np.zeros(n_cons, dtype=np.int8)

        # Bounds
        lb = np.zeros(n_vars)
        ub = capacities

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"network_{self.instance_counter}",
        )

    def _generate_blending(self) -> LPInstance:
        """
        Generate a blending/mixing problem.

        min sum_i cost_i * x_i
        s.t. sum_i content_ij * x_i >= min_j  (minimum content)
             sum_i content_ij * x_i <= max_j  (maximum content)
             sum_i x_i = total_amount
             x_i >= 0

        Modified to ensure feasibility by generating a feasible blend first.
        """
        n_ingredients = self.rng.randint(5, 30)
        n_properties = self.rng.randint(2, 5)  # Fewer properties for easier feasibility

        n_vars = n_ingredients
        n_cons = 2 * n_properties + 1  # min, max for each property + total

        # Cost per unit
        c = self.rng.uniform(1, 100, n_vars)

        # Content matrix - each ingredient has some of each property
        content = self.rng.uniform(5, 95, (n_properties, n_ingredients))

        # Generate a feasible blend: uniform mixture
        total_amount = self.rng.uniform(100, 1000)
        x_feasible = np.full(n_ingredients, total_amount / n_ingredients)

        # Compute achieved property values from the feasible blend
        achieved = content @ x_feasible

        # Set bounds around achieved values with some slack
        min_content = achieved * self.rng.uniform(0.5, 0.8, n_properties)
        max_content = achieved * self.rng.uniform(1.2, 1.5, n_properties)

        # Build constraint matrix
        A_data = []
        A_indices = []
        A_indptr = [0]

        # Minimum content constraints
        for p in range(n_properties):
            for i in range(n_ingredients):
                A_data.append(content[p, i])
                A_indices.append(i)
            A_indptr.append(len(A_data))

        # Maximum content constraints
        for p in range(n_properties):
            for i in range(n_ingredients):
                A_data.append(content[p, i])
                A_indices.append(i)
            A_indptr.append(len(A_data))

        # Total amount constraint
        for i in range(n_ingredients):
            A_data.append(1.0)
            A_indices.append(i)
        A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS
        b = np.concatenate([min_content, max_content, [total_amount]])

        # Sense: >= for min, <= for max, = for total
        sense = np.concatenate([
            np.full(n_properties, 1, dtype=np.int8),
            np.full(n_properties, -1, dtype=np.int8),
            np.array([0], dtype=np.int8)
        ])

        # Bounds
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, total_amount)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"blending_{self.instance_counter}",
        )

    def _generate_production_planning(self) -> LPInstance:
        """
        Generate a multi-period production planning problem.

        min sum_pt (prod_cost * x_pt + inv_cost * inv_pt)
        s.t. inv_{p,t-1} + x_pt - inv_pt = demand_pt  (balance)
             sum_p resource_p * x_pt <= capacity_t  (resource)
             x_pt, inv_pt >= 0

        Modified: ensure feasibility by setting capacity high enough.
        """
        n_products = self.rng.randint(3, 10)
        n_periods = self.rng.randint(5, 20)

        # Variables: x_pt (production), inv_pt (inventory)
        n_vars = 2 * n_products * n_periods
        # Constraints: balance, capacity
        n_cons = n_products * n_periods + n_periods

        # Indices
        def x_idx(p, t):
            return p * n_periods + t

        def inv_idx(p, t):
            return n_products * n_periods + p * n_periods + t

        # Costs
        c = np.zeros(n_vars)
        for p in range(n_products):
            prod_cost = self.rng.uniform(10, 100)
            inv_cost = self.rng.uniform(1, 10)
            for t in range(n_periods):
                c[x_idx(p, t)] = prod_cost
                c[inv_idx(p, t)] = inv_cost

        # Generate demands first
        demands = self.rng.uniform(10, 50, (n_products, n_periods))

        # Resource usage per product
        resource_usage = self.rng.uniform(0.5, 1.5, n_products)

        # Compute minimum capacity needed at each period (all demand met by production)
        min_capacity = np.sum(demands * resource_usage[:, np.newaxis], axis=0)

        # Set capacity with slack
        capacities = min_capacity * self.rng.uniform(1.2, 2.0, n_periods)

        # Build constraints
        A_data = []
        A_indices = []
        A_indptr = [0]

        # Balance constraints
        for p in range(n_products):
            for t in range(n_periods):
                # x_pt: +1
                A_data.append(1.0)
                A_indices.append(x_idx(p, t))

                # inv_pt: -1
                A_data.append(-1.0)
                A_indices.append(inv_idx(p, t))

                # inv_{p,t-1}: +1
                if t > 0:
                    A_data.append(1.0)
                    A_indices.append(inv_idx(p, t - 1))

                A_indptr.append(len(A_data))

        # Capacity constraints
        for t in range(n_periods):
            for p in range(n_products):
                A_data.append(resource_usage[p])
                A_indices.append(x_idx(p, t))
            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS - flatten demands
        b = np.concatenate([demands.flatten(), capacities])

        # Sense
        sense = np.concatenate([
            np.zeros(n_products * n_periods, dtype=np.int8),  # balance: =
            np.full(n_periods, -1, dtype=np.int8)  # capacity: <=
        ])

        # Bounds
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"production_{self.instance_counter}",
        )

    def _generate_generic(self) -> LPInstance:
        """Generate a generic random LP."""
        n_vars, n_cons = self._random_size()
        density = self._random_density()

        # Objective
        c = self._random_coefficients(n_vars)

        # Build sparse matrix
        nnz = max(1, int(n_vars * n_cons * density))

        A_data = []
        A_indices = []
        A_indptr = [0]

        for i in range(n_cons):
            # Number of non-zeros in this row
            row_nnz = max(1, self.rng.poisson(nnz / n_cons))
            row_nnz = min(row_nnz, n_vars)

            cols = self.rng.choice(n_vars, size=row_nnz, replace=False)
            cols.sort()

            for j in cols:
                A_data.append(self._random_coefficients(1)[0])
                A_indices.append(j)

            A_indptr.append(len(A_data))

        A_data = np.array(A_data)
        A_indices = np.array(A_indices, dtype=np.int32)
        A_indptr = np.array(A_indptr, dtype=np.int32)

        # RHS
        b = self.rng.uniform(0, self.config.rhs_scale, n_cons)

        # Sense
        sense = self._random_sense(n_cons)

        # Bounds
        lb, ub = self._random_bounds(n_vars)

        return LPInstance(
            c=c,
            A_data=A_data,
            A_indices=A_indices,
            A_indptr=A_indptr,
            b=b,
            lb=lb,
            ub=ub,
            sense=sense,
            name=f"generic_{self.instance_counter}",
        )


def create_corpus(
    n_train: int = 100,
    n_valid: int = 20,
    n_test: int = 30,
    seed: int = 42,
    config: Optional[GeneratorConfig] = None,
) -> Dict[str, List[LPInstance]]:
    """
    Create train/valid/test split of LP instances.

    Args:
        n_train: Number of training instances
        n_valid: Number of validation instances
        n_test: Number of test instances
        seed: Random seed for reproducibility
        config: Generator configuration

    Returns:
        Dictionary with 'train', 'valid', 'test' keys
    """
    gen = LPGenerator(seed=seed, config=config)

    return {
        'train': gen.generate_batch(n_train),
        'valid': gen.generate_batch(n_valid),
        'test': gen.generate_batch(n_test),
    }


# Public LP benchmark instances (instructions for download)
PUBLIC_INSTANCES = [
    {
        "name": "afiro",
        "source": "netlib",
        "url": "https://www.netlib.org/lp/data/afiro",
        "description": "Small test problem",
    },
    {
        "name": "blend",
        "source": "netlib",
        "url": "https://www.netlib.org/lp/data/blend",
        "description": "Blending problem",
    },
    {
        "name": "kb2",
        "source": "netlib",
        "url": "https://www.netlib.org/lp/data/kb2",
        "description": "Small problem",
    },
    {
        "name": "sc50a",
        "source": "netlib",
        "url": "https://www.netlib.org/lp/data/sc50a",
        "description": "Set covering",
    },
    {
        "name": "sc50b",
        "source": "netlib",
        "url": "https://www.netlib.org/lp/data/sc50b",
        "description": "Set covering",
    },
]


def get_public_instance_info() -> List[Dict]:
    """Get information about available public LP instances."""
    return PUBLIC_INSTANCES
