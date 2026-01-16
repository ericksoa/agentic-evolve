#!/usr/bin/env python3
"""
create_benchmarks.py - Create reproducible LP benchmark problems.

Generates realistic LP problems from various domains that serve as
a grounded benchmark for comparing cuOpt configurations.

These problems are derived from real-world optimization domains:
- Transportation and logistics
- Production planning
- Network flow
- Portfolio optimization
- Resource allocation

Usage:
    python3 -m showcase.cuopt_lp_autotuner.create_benchmarks --output ./benchmarks
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import argparse
from scipy.sparse import csr_matrix

from .mps_loader import LPData


def create_transportation_lp(n_sources: int, n_destinations: int, seed: int) -> LPData:
    """
    Create a transportation problem.

    min sum_ij c_ij * x_ij
    s.t. sum_j x_ij <= supply_i   (supply constraints)
         sum_i x_ij >= demand_j   (demand constraints)
         x_ij >= 0
    """
    rng = np.random.RandomState(seed)
    n_vars = n_sources * n_destinations
    n_cons = n_sources + n_destinations

    # Cost coefficients (distance-based)
    source_locs = rng.uniform(0, 100, (n_sources, 2))
    dest_locs = rng.uniform(0, 100, (n_destinations, 2))
    c = np.zeros(n_vars)
    for i in range(n_sources):
        for j in range(n_destinations):
            dist = np.linalg.norm(source_locs[i] - dest_locs[j])
            c[i * n_destinations + j] = dist * rng.uniform(0.8, 1.2)

    # Build constraint matrix
    rows, cols, data = [], [], []

    # Supply constraints: sum_j x_ij <= supply_i
    for i in range(n_sources):
        for j in range(n_destinations):
            rows.append(i)
            cols.append(i * n_destinations + j)
            data.append(1.0)

    # Demand constraints: sum_i x_ij >= demand_j
    for j in range(n_destinations):
        for i in range(n_sources):
            rows.append(n_sources + j)
            cols.append(i * n_destinations + j)
            data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))

    # RHS: supplies and demands (balanced)
    total_demand = rng.uniform(1000, 10000)
    supplies = rng.dirichlet(np.ones(n_sources)) * total_demand * 1.1
    demands = rng.dirichlet(np.ones(n_destinations)) * total_demand

    b = np.concatenate([supplies, demands])
    sense = np.concatenate([
        np.full(n_sources, -1),    # <=
        np.full(n_destinations, 1)  # >=
    ]).astype(np.int8)

    return LPData(
        c=c,
        A=A,
        b=b,
        sense=sense,
        lb=np.zeros(n_vars),
        ub=np.full(n_vars, np.inf),
        name=f"transport_{n_sources}x{n_destinations}",
    )


def create_production_planning_lp(n_products: int, n_periods: int, seed: int) -> LPData:
    """
    Create a multi-period production planning problem.

    Variables: x[p,t] = production, s[p,t] = inventory
    min sum_pt (prod_cost * x[p,t] + hold_cost * s[p,t])
    s.t. s[p,t-1] + x[p,t] - s[p,t] = demand[p,t]  (balance)
         sum_p resource[p] * x[p,t] <= capacity[t]  (capacity)
         x, s >= 0
    """
    rng = np.random.RandomState(seed)

    n_vars = 2 * n_products * n_periods  # production + inventory
    n_cons = n_products * n_periods + n_periods  # balance + capacity

    def x_idx(p, t):
        return p * n_periods + t

    def s_idx(p, t):
        return n_products * n_periods + p * n_periods + t

    # Costs
    c = np.zeros(n_vars)
    prod_costs = rng.uniform(10, 50, n_products)
    hold_costs = rng.uniform(1, 5, n_products)
    for p in range(n_products):
        for t in range(n_periods):
            c[x_idx(p, t)] = prod_costs[p]
            c[s_idx(p, t)] = hold_costs[p]

    # Generate demands
    base_demand = rng.uniform(20, 100, n_products)
    seasonality = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / n_periods)
    demands = np.outer(base_demand, seasonality) * rng.uniform(0.8, 1.2, (n_products, n_periods))

    # Resource usage and capacity
    resource_usage = rng.uniform(0.5, 2.0, n_products)
    min_capacity = np.max(np.sum(demands * resource_usage[:, np.newaxis], axis=0))
    capacities = min_capacity * rng.uniform(1.2, 1.5, n_periods)

    # Build constraints
    rows, cols, data = [], [], []
    b = []
    sense = []

    # Balance constraints: s[p,t-1] + x[p,t] - s[p,t] = demand[p,t]
    con_idx = 0
    for p in range(n_products):
        for t in range(n_periods):
            # x[p,t]: +1
            rows.append(con_idx)
            cols.append(x_idx(p, t))
            data.append(1.0)

            # s[p,t]: -1
            rows.append(con_idx)
            cols.append(s_idx(p, t))
            data.append(-1.0)

            # s[p,t-1]: +1
            if t > 0:
                rows.append(con_idx)
                cols.append(s_idx(p, t - 1))
                data.append(1.0)

            b.append(demands[p, t])
            sense.append(0)  # =
            con_idx += 1

    # Capacity constraints: sum_p resource[p] * x[p,t] <= capacity[t]
    for t in range(n_periods):
        for p in range(n_products):
            rows.append(con_idx)
            cols.append(x_idx(p, t))
            data.append(resource_usage[p])

        b.append(capacities[t])
        sense.append(-1)  # <=
        con_idx += 1

    A = csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))

    return LPData(
        c=c,
        A=A,
        b=np.array(b),
        sense=np.array(sense, dtype=np.int8),
        lb=np.zeros(n_vars),
        ub=np.full(n_vars, np.inf),
        name=f"production_{n_products}x{n_periods}",
    )


def create_portfolio_lp(n_assets: int, seed: int) -> LPData:
    """
    Create a portfolio optimization problem (LP relaxation).

    max sum_i return_i * x_i
    s.t. sum_i x_i = 1                    (budget)
         sum_i risk_i * x_i <= max_risk   (risk limit)
         x_i >= min_alloc                 (diversification)
         x_i <= max_alloc                 (concentration limit)
    """
    rng = np.random.RandomState(seed)

    n_vars = n_assets
    n_cons = 2  # budget + risk

    # Returns (maximize -> negate for minimization)
    returns = rng.uniform(0.02, 0.15, n_assets)
    c = -returns

    # Risk scores
    risks = rng.uniform(0.1, 0.5, n_assets)

    # Constraints
    rows, cols, data = [], [], []

    # Budget: sum x_i = 1
    for i in range(n_assets):
        rows.append(0)
        cols.append(i)
        data.append(1.0)

    # Risk: sum risk_i * x_i <= max_risk
    for i in range(n_assets):
        rows.append(1)
        cols.append(i)
        data.append(risks[i])

    A = csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))
    b = np.array([1.0, 0.3])  # budget=1, max_risk=0.3
    sense = np.array([0, -1], dtype=np.int8)  # =, <=

    # Bounds for diversification
    lb = np.full(n_vars, 0.01)  # min 1% per asset
    ub = np.full(n_vars, 0.20)  # max 20% per asset

    return LPData(
        c=c,
        A=A,
        b=b,
        sense=sense,
        lb=lb,
        ub=ub,
        name=f"portfolio_{n_assets}",
    )


def create_network_flow_lp(n_nodes: int, n_edges: int, seed: int) -> LPData:
    """
    Create a minimum cost network flow problem.

    min sum_e cost_e * x_e
    s.t. flow conservation at each node
         0 <= x_e <= capacity_e
    """
    rng = np.random.RandomState(seed)

    # Generate random connected graph
    edges = []
    # Spanning tree for connectivity
    for i in range(1, n_nodes):
        j = rng.randint(0, i)
        edges.append((min(j, i), max(j, i)))

    # Add random edges
    edge_set = set(edges)
    while len(edges) < n_edges:
        i, j = rng.randint(0, n_nodes, 2)
        if i != j:
            e = (min(i, j), max(i, j))
            if e not in edge_set:
                edges.append(e)
                edge_set.add(e)

    n_vars = len(edges)
    n_cons = n_nodes

    # Costs
    c = rng.uniform(1, 20, n_vars)

    # Capacities and feasible flow
    capacities = rng.uniform(50, 200, n_vars)
    feasible_flow = capacities * rng.uniform(0.3, 0.7, n_vars)

    # Compute supply/demand from feasible flow
    supply = np.zeros(n_nodes)
    for e_idx, (i, j) in enumerate(edges):
        # Flow goes from i to j
        supply[i] -= feasible_flow[e_idx]
        supply[j] += feasible_flow[e_idx]

    # Build incidence matrix
    rows, cols, data = [], [], []
    for v in range(n_nodes):
        for e_idx, (i, j) in enumerate(edges):
            if i == v:
                rows.append(v)
                cols.append(e_idx)
                data.append(-1.0)  # outgoing
            elif j == v:
                rows.append(v)
                cols.append(e_idx)
                data.append(1.0)   # incoming

    A = csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))

    return LPData(
        c=c,
        A=A,
        b=supply,
        sense=np.zeros(n_cons, dtype=np.int8),  # all =
        lb=np.zeros(n_vars),
        ub=capacities,
        name=f"network_{n_nodes}n_{n_vars}e",
    )


def create_resource_allocation_lp(n_jobs: int, n_resources: int, seed: int) -> LPData:
    """
    Create a resource allocation problem.

    max sum_j profit_j * x_j
    s.t. sum_j usage_ij * x_j <= available_i  (resource limits)
         x_j >= 0, x_j <= max_j
    """
    rng = np.random.RandomState(seed)

    n_vars = n_jobs
    n_cons = n_resources

    # Profits (maximize -> negate)
    profits = rng.uniform(10, 100, n_jobs)
    c = -profits

    # Resource usage (sparse)
    density = min(0.3, 3.0 / n_jobs)
    rows, cols, data = [], [], []
    for r in range(n_resources):
        for j in range(n_jobs):
            if rng.random() < density:
                rows.append(r)
                cols.append(j)
                data.append(rng.uniform(1, 10))

    A = csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))

    # Available resources
    b = rng.uniform(100, 500, n_resources)
    sense = np.full(n_cons, -1, dtype=np.int8)  # all <=

    # Bounds
    lb = np.zeros(n_vars)
    ub = rng.uniform(10, 50, n_vars)

    return LPData(
        c=c,
        A=A,
        b=b,
        sense=sense,
        lb=lb,
        ub=ub,
        name=f"resource_{n_jobs}j_{n_resources}r",
    )


# Benchmark suite definition
BENCHMARK_SUITE = [
    # Small problems (quick solve)
    {"type": "transportation", "params": {"n_sources": 5, "n_destinations": 8}, "seed": 1001},
    {"type": "transportation", "params": {"n_sources": 10, "n_destinations": 15}, "seed": 1002},
    {"type": "portfolio", "params": {"n_assets": 20}, "seed": 2001},
    {"type": "portfolio", "params": {"n_assets": 50}, "seed": 2002},
    {"type": "resource", "params": {"n_jobs": 30, "n_resources": 10}, "seed": 3001},

    # Medium problems
    {"type": "production", "params": {"n_products": 5, "n_periods": 20}, "seed": 4001},
    {"type": "production", "params": {"n_products": 10, "n_periods": 30}, "seed": 4002},
    {"type": "network", "params": {"n_nodes": 30, "n_edges": 80}, "seed": 5001},
    {"type": "network", "params": {"n_nodes": 50, "n_edges": 150}, "seed": 5002},
    {"type": "transportation", "params": {"n_sources": 20, "n_destinations": 30}, "seed": 1003},

    # Larger problems
    {"type": "resource", "params": {"n_jobs": 100, "n_resources": 20}, "seed": 3002},
    {"type": "production", "params": {"n_products": 15, "n_periods": 50}, "seed": 4003},
    {"type": "network", "params": {"n_nodes": 80, "n_edges": 250}, "seed": 5003},
    {"type": "portfolio", "params": {"n_assets": 100}, "seed": 2003},
    {"type": "transportation", "params": {"n_sources": 30, "n_destinations": 40}, "seed": 1004},
]


def create_benchmark(spec: Dict[str, Any]) -> LPData:
    """Create a benchmark problem from spec."""
    lp_type = spec["type"]
    params = spec["params"]
    seed = spec["seed"]

    if lp_type == "transportation":
        return create_transportation_lp(seed=seed, **params)
    elif lp_type == "production":
        return create_production_planning_lp(seed=seed, **params)
    elif lp_type == "portfolio":
        return create_portfolio_lp(seed=seed, **params)
    elif lp_type == "network":
        return create_network_flow_lp(seed=seed, **params)
    elif lp_type == "resource":
        return create_resource_allocation_lp(seed=seed, **params)
    else:
        raise ValueError(f"Unknown problem type: {lp_type}")


def save_benchmark(lp: LPData, output_dir: Path) -> str:
    """Save benchmark as JSON (portable format)."""
    filepath = output_dir / f"{lp.name}.json"

    data = {
        "name": lp.name,
        "n_vars": lp.n_vars,
        "n_cons": lp.n_cons,
        "c": lp.c.tolist(),
        "A_data": lp.A.data.tolist(),
        "A_indices": lp.A.indices.tolist(),
        "A_indptr": lp.A.indptr.tolist(),
        "b": lp.b.tolist(),
        "sense": lp.sense.tolist(),
        "lb": lp.lb.tolist(),
        "ub": [float(x) if np.isfinite(x) else None for x in lp.ub],
    }

    with open(filepath, 'w') as f:
        json.dump(data, f)

    return str(filepath)


def load_benchmark(filepath: str) -> LPData:
    """Load benchmark from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    A = csr_matrix(
        (data["A_data"], data["A_indices"], data["A_indptr"]),
        shape=(data["n_cons"], data["n_vars"])
    )

    ub = np.array([x if x is not None else np.inf for x in data["ub"]])

    return LPData(
        c=np.array(data["c"]),
        A=A,
        b=np.array(data["b"]),
        sense=np.array(data["sense"], dtype=np.int8),
        lb=np.array(data["lb"]),
        ub=ub,
        name=data["name"],
    )


def create_all_benchmarks(output_dir: str, verbose: bool = True) -> List[str]:
    """Create all benchmark problems."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Creating {len(BENCHMARK_SUITE)} benchmark problems in {output_path}/")
        print()
        print(f"{'Name':<30} {'Type':<15} {'Vars':>8} {'Cons':>8}")
        print("-" * 65)

    files = []
    for spec in BENCHMARK_SUITE:
        lp = create_benchmark(spec)
        filepath = save_benchmark(lp, output_path)
        files.append(filepath)

        if verbose:
            print(f"{lp.name:<30} {spec['type']:<15} {lp.n_vars:>8} {lp.n_cons:>8}")

    if verbose:
        print("-" * 65)
        print(f"Created {len(files)} benchmarks")

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible LP benchmark problems"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./benchmarks",
        help="Output directory (default: ./benchmarks)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List benchmark specs and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Benchmark Suite:")
        print()
        for i, spec in enumerate(BENCHMARK_SUITE):
            print(f"{i+1:2}. {spec['type']:<15} {spec['params']} (seed={spec['seed']})")
        return

    create_all_benchmarks(args.output)


if __name__ == "__main__":
    main()
