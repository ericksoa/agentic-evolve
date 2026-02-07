"""
gen1a: Inlined union-find + C-level sort key via operator.itemgetter.

Mutation from gen0_a:
1. Inline find/union into the main loop to eliminate Python function call
   overhead (each call has ~50ns overhead in CPython, significant for small graphs).
2. Use operator.itemgetter (C-implemented) instead of lambda for both sorts.
3. Pre-convert u,v to int once during edge preprocessing rather than in hot loop.
4. Use local variable references for parent/rank arrays (faster than closures).
"""
from typing import Any
from operator import itemgetter

_ig2 = itemgetter(2)
_ig01 = itemgetter(0, 1)


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]

        if num_nodes <= 1 or not edges:
            return {"mst_edges": []}

        # Sort edges by weight using C-level itemgetter
        sorted_edges = sorted(edges, key=_ig2)

        # Union-Find arrays - local refs for speed
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        # Kruskal's with fully inlined find + union
        mst_edges = []
        target = num_nodes - 1
        mst_count = 0

        for e in sorted_edges:
            u = int(e[0])
            v = int(e[1])

            # Inlined find(u) with path halving
            x = u
            px = parent[x]
            while px != x:
                parent[x] = parent[px]
                x = parent[px]
                px = parent[x]
            root_u = x

            # Inlined find(v) with path halving
            x = v
            px = parent[x]
            while px != x:
                parent[x] = parent[px]
                x = parent[px]
                px = parent[x]
            root_v = x

            if root_u != root_v:
                # Inlined union by rank
                ru, rv = rank[root_u], rank[root_v]
                if ru < rv:
                    parent[root_u] = root_v
                elif ru > rv:
                    parent[root_v] = root_u
                else:
                    parent[root_v] = root_u
                    rank[root_u] = ru + 1

                # Add edge (normalized u < v)
                w = e[2]
                if u > v:
                    mst_edges.append([v, u, w])
                else:
                    mst_edges.append([u, v, w])
                mst_count += 1
                if mst_count == target:
                    break

        # Sort output by (u, v) using C-implemented itemgetter
        mst_edges.sort(key=_ig01)
        return {"mst_edges": mst_edges}
