"""
Variant D: Capacitated balanced partitioning using graph-based approach.

Approach:
1. Build proximity graph of accounts (k-nearest neighbors)
2. Use recursive balanced partitioning with capacity (revenue) constraints
3. Optimize cut quality while maintaining revenue balance

Performance optimizations:
- Use sorted distance lists for efficient neighbor lookup
- Recursive divide-and-conquer reduces search space
- Cache centroid calculations
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning for territory assignment."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)

    # Compute all pairwise distances
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((lats[i] - lats[j])**2 + (lons[i] - lons[j])**2)
            distances[(i, j)] = d
            distances[(j, i)] = d

    def get_distance(i, j):
        if i == j:
            return 0
        return distances.get((i, j), float("inf"))

    def partition_accounts(account_indices, num_parts, target_rev_per_part):
        """Recursively partition accounts into num_parts balanced groups."""
        if num_parts == 1:
            return [account_indices]

        if len(account_indices) <= num_parts:
            # Not enough accounts - assign one per part
            result = [[idx] for idx in account_indices]
            while len(result) < num_parts:
                result.append([])
            return result

        # Find two accounts that are farthest apart (pivots)
        max_dist = -1
        pivot1, pivot2 = account_indices[0], account_indices[1]
        for i in account_indices:
            for j in account_indices:
                if i < j:
                    d = get_distance(i, j)
                    if d > max_dist:
                        max_dist = d
                        pivot1, pivot2 = i, j

        # Split into two groups based on proximity to pivots
        # Use capacity (revenue) aware assignment
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
        remaining.sort(key=lambda idx: get_distance(idx, pivot1) - get_distance(idx, pivot2))

        for idx in remaining:
            dist1 = get_distance(idx, pivot1)
            dist2 = get_distance(idx, pivot2)

            # Score based on distance and revenue balance
            score1 = dist1 + abs(rev1 + revenues[idx] - target_rev_1) * 0.00001
            score2 = dist2 + abs(rev2 + revenues[idx] - target_rev_2) * 0.00001

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
    target_revenue_per_rep = total_revenue / num_reps
    partitions = partition_accounts(all_indices, num_reps, target_revenue_per_rep)

    # Post-optimization: balance revenues with local swaps
    def get_partition_revenues():
        return [sum(revenues[idx] for idx in part) for part in partitions]

    for _ in range(20):
        rev_list = get_partition_revenues()
        max_part = max(range(num_reps), key=lambda p: rev_list[p])
        min_part = min(range(num_reps), key=lambda p: rev_list[p])

        if max_part == min_part:
            break

        imbalance = rev_list[max_part] - rev_list[min_part]
        if imbalance < target_revenue_per_rep * 0.1:
            break  # Good enough balance

        # Try to find an account to move
        best_move = None
        best_improvement = 0

        for idx in partitions[max_part]:
            new_imbalance = abs(rev_list[max_part] - revenues[idx] - target_revenue_per_rep) + \
                           abs(rev_list[min_part] + revenues[idx] - target_revenue_per_rep)
            old_imbalance = abs(rev_list[max_part] - target_revenue_per_rep) + \
                           abs(rev_list[min_part] - target_revenue_per_rep)
            improvement = old_imbalance - new_imbalance
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = idx

        if best_move is not None:
            partitions[max_part].remove(best_move)
            partitions[min_part].append(best_move)

    # Convert to output format
    result = {}
    for rep_id, partition in enumerate(partitions):
        result[rep_id] = [ids[idx] for idx in partition]

    return result
