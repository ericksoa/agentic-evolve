"""
Variant gen7c: Optimized balancing with set-based operations and reduced allocations.

Parent: gen1b (fitness 0.7993)

Mutation type: Cache optimization + Branch elimination
- Replace list.remove() O(n) operations with set-based O(1) operations
- Use index-based partition tracking instead of list membership lookups
- Reduce temporary list allocations in the hot balancing loop
- Precompute revenue deviations to avoid repeated abs() calls

Hypothesis: The parent's balancing loop uses list.remove() which is O(n) and
creates temporary list comprehensions. By using sets for partition membership
and tracking revenues incrementally, we reduce the per-iteration cost
significantly, enabling more balancing iterations and better final balance.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with optimized set operations."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue_per_rep = total_revenue / num_reps

    # Precompute all pairwise squared distances (avoid sqrt during lookup)
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2
            dist_sq[i][j] = d_sq
            dist_sq[j][i] = d_sq

    # Precompute sorted neighbor lists for each account (nearest to farthest)
    neighbors = []
    for i in range(n):
        neighbor_list = [(j, dist_sq[i][j]) for j in range(n) if j != i]
        neighbor_list.sort(key=lambda x: x[1])
        neighbors.append([x[0] for x in neighbor_list])

    def partition_accounts(account_indices, num_parts, target_rev_per_part):
        """Recursively partition accounts into num_parts balanced groups."""
        if num_parts == 1:
            return [list(account_indices)]

        if len(account_indices) <= num_parts:
            # Not enough accounts - assign one per part
            result = [[idx] for idx in account_indices]
            while len(result) < num_parts:
                result.append([])
            return result

        # Create index set for O(1) membership check
        idx_set = set(account_indices)

        # Find two accounts that are farthest apart (pivots)
        max_dist_sq = -1
        pivot1, pivot2 = account_indices[0], account_indices[1]

        for i in account_indices:
            # Check farthest neighbor that's in our subset
            for far_j in reversed(neighbors[i]):
                if far_j in idx_set:
                    d = dist_sq[i][far_j]
                    if d > max_dist_sq:
                        max_dist_sq = d
                        pivot1, pivot2 = i, far_j
                    break

        # Split into two groups based on proximity to pivots
        parts_1 = num_parts // 2
        parts_2 = num_parts - parts_1
        target_rev_1 = target_rev_per_part * parts_1
        target_rev_2 = target_rev_per_part * parts_2

        group1 = [pivot1]
        group2 = [pivot2]
        rev1 = revenues[pivot1]
        rev2 = revenues[pivot2]

        remaining = [idx for idx in account_indices if idx not in (pivot1, pivot2)]

        # Sort remaining by distance difference to pivots
        remaining.sort(key=lambda idx: dist_sq[idx][pivot1] - dist_sq[idx][pivot2])

        for idx in remaining:
            d1 = dist_sq[idx][pivot1]
            d2 = dist_sq[idx][pivot2]

            # Score based on distance and revenue balance
            score1 = d1 + abs(rev1 + revenues[idx] - target_rev_1) * 0.00001
            score2 = d2 + abs(rev2 + revenues[idx] - target_rev_2) * 0.00001

            # Bias towards balance if one group is overfilled
            if rev1 > target_rev_1 * 1.2:
                score1 *= 2
            if rev2 > target_rev_2 * 1.2:
                score2 *= 2

            if score1 <= score2:
                group1.append(idx)
                rev1 += revenues[idx]
            else:
                group2.append(idx)
                rev2 += revenues[idx]

        # Recursively partition each group
        result1 = partition_accounts(group1, parts_1, target_rev_per_part)
        result2 = partition_accounts(group2, parts_2, target_rev_per_part)

        return result1 + result2

    # Initial partitioning
    all_indices = list(range(n))
    partitions = partition_accounts(all_indices, num_reps, target_revenue_per_rep)

    # MUTATION: Convert partitions to sets for O(1) add/remove operations
    partition_sets = [set(p) for p in partitions]
    rep_revenues = [sum(revenues[idx] for idx in p) for p in partitions]

    # Track which partition each account belongs to for O(1) lookup
    assignments = [0] * n
    for rep_id, part in enumerate(partition_sets):
        for idx in part:
            assignments[idx] = rep_id

    # MUTATION: Optimized balancing loop with set operations
    for _ in range(30):  # Increased iterations since each is cheaper
        # Find max/min partitions in single pass
        max_part = 0
        min_part = 0
        max_rev = rep_revenues[0]
        min_rev = rep_revenues[0]

        for p in range(1, num_reps):
            if rep_revenues[p] > max_rev:
                max_rev = rep_revenues[p]
                max_part = p
            if rep_revenues[p] < min_rev:
                min_rev = rep_revenues[p]
                min_part = p

        if max_part == min_part:
            break

        imbalance = max_rev - min_rev
        if imbalance < target_revenue_per_rep * 0.1:
            break

        # Try to find an account to move
        best_move = None
        best_combined_score = 0

        # MUTATION: Precompute target deviations to avoid repeated calculation
        max_part_target_dev = abs(max_rev - target_revenue_per_rep)
        min_part_target_dev = abs(min_rev - target_revenue_per_rep)
        old_total_dev = max_part_target_dev + min_part_target_dev

        # MUTATION: Cache partition list to avoid set iteration overhead
        max_part_list = list(partition_sets[max_part])

        for idx in max_part_list:
            rev_idx = revenues[idx]
            new_max_dev = abs(max_rev - rev_idx - target_revenue_per_rep)
            new_min_dev = abs(min_rev + rev_idx - target_revenue_per_rep)
            improvement = old_total_dev - (new_max_dev + new_min_dev)

            # Compactness bonus: prefer moving to nearby accounts in min_part
            if partition_sets[min_part]:
                # MUTATION: Use precomputed distances with early exit for large partitions
                min_dist_to_target = float('inf')
                for other in partition_sets[min_part]:
                    d = dist_sq[idx][other]
                    if d < min_dist_to_target:
                        min_dist_to_target = d
                        # Early exit if we find a very close neighbor
                        if d < 0.01:
                            break
                compactness_bonus = 1.0 / (1.0 + min_dist_to_target)
            else:
                compactness_bonus = 0

            # Combined score favoring both balance and compactness
            combined = improvement + compactness_bonus * 0.01

            if combined > best_combined_score:
                best_combined_score = combined
                best_move = idx

        if best_move is not None and best_combined_score > 0:
            # MUTATION: O(1) set operations instead of O(n) list operations
            partition_sets[max_part].remove(best_move)
            partition_sets[min_part].add(best_move)
            rep_revenues[max_part] -= revenues[best_move]
            rep_revenues[min_part] += revenues[best_move]
            assignments[best_move] = min_part

    # Convert to output format using assignments array (O(n))
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
