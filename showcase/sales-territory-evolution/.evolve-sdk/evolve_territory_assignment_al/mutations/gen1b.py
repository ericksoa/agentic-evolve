"""
Variant gen1b: Precomputed sorted neighbor lists for faster partitioning.

Mutation from gen0_d: Cache optimization - Replace O(n^2) pivot search in each
recursive call with precomputed sorted neighbor lists. This improves cache
locality and reduces redundant distance calculations.

Key changes:
1. Precompute sorted neighbor lists for each account (by distance)
2. Use neighbor lists to find antipodal points (pivots) more efficiently
3. Improve the local swap phase by prioritizing nearby accounts

Hypothesis: Precomputed neighbor lists eliminate redundant distance
calculations in recursive calls and enable smarter swap candidates,
improving both performance and compactness.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with precomputed neighbor lists."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)

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

    # Precompute farthest neighbor for each account (for fast pivot selection)
    farthest = [neighbors[i][-1] if neighbors[i] else i for i in range(n)]

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

        # Create index set for O(1) membership check
        idx_set = set(account_indices)

        # Find two accounts that are farthest apart (pivots)
        # Use precomputed farthest neighbors for efficiency
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
                    break  # Only need to check one (farthest in set)

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
    target_revenue_per_rep = total_revenue / num_reps
    partitions = partition_accounts(all_indices, num_reps, target_revenue_per_rep)

    # Post-optimization: balance revenues with local swaps
    # Enhanced: prioritize swapping geographically nearby accounts
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
        best_compactness_bonus = 0

        for idx in partitions[max_part]:
            new_imbalance = abs(rev_list[max_part] - revenues[idx] - target_revenue_per_rep) + \
                           abs(rev_list[min_part] + revenues[idx] - target_revenue_per_rep)
            old_imbalance = abs(rev_list[max_part] - target_revenue_per_rep) + \
                           abs(rev_list[min_part] - target_revenue_per_rep)
            improvement = old_imbalance - new_imbalance

            # Compactness bonus: prefer moving to nearby accounts in min_part
            if partitions[min_part]:
                min_dist_to_target = min(dist_sq[idx][other] for other in partitions[min_part])
                compactness_bonus = 1.0 / (1.0 + min_dist_to_target)
            else:
                compactness_bonus = 0

            # Combined score favoring both balance and compactness
            combined = improvement + compactness_bonus * 0.01

            if combined > best_improvement:
                best_improvement = combined
                best_move = idx
                best_compactness_bonus = compactness_bonus

        if best_move is not None and best_improvement > 0:
            partitions[max_part].remove(best_move)
            partitions[min_part].append(best_move)

    # Convert to output format
    result = {}
    for rep_id, partition in enumerate(partitions):
        result[rep_id] = [ids[idx] for idx in partition]

    return result
