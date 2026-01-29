"""
Variant gen6c: Optimized pivot search with direct O(n^2) scan.

Parent: gen1b (0.7993)

Mutation: Branch elimination / Loop optimization
- Replace the pivot search using reversed(neighbors[i]) with set membership checks
  with a direct O(n^2) scan within the subset
- The original approach iterates through reversed neighbor lists looking for the
  first neighbor in the subset set - this has unpredictable branching and set
  lookups in the inner loop
- For small-to-medium subsets, a direct O(|subset|^2) scan with array indexing
  is more cache-friendly and avoids set hashing overhead
- Also use a local variable for dist_sq access to reduce repeated list lookups

Hypothesis: The original pivot search has poor branch prediction (breaks early on
variable conditions) and set membership checks in the inner loop. A direct scan
with array indexing has more predictable memory access patterns and eliminates
the set overhead. For typical recursive partitioning depths, the subset sizes
are manageable and the cache-friendly access pattern should win.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with optimized pivot search."""
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
        lat_i = lats[i]
        lon_i = lons[i]
        dist_sq_i = dist_sq[i]
        for j in range(i + 1, n):
            d_sq = (lat_i - lats[j])**2 + (lon_i - lons[j])**2
            dist_sq_i[j] = d_sq
            dist_sq[j][i] = d_sq

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

        # MUTATION: Direct O(n^2) scan for farthest pair (branch elimination)
        # This replaces the neighbor list traversal with set membership checks
        max_dist_sq = -1
        pivot1, pivot2 = account_indices[0], account_indices[1] if len(account_indices) > 1 else account_indices[0]

        # Direct scan through all pairs in the subset
        num_indices = len(account_indices)
        for ii in range(num_indices):
            i = account_indices[ii]
            dist_sq_i = dist_sq[i]  # Cache row reference
            for jj in range(ii + 1, num_indices):
                j = account_indices[jj]
                d = dist_sq_i[j]
                if d > max_dist_sq:
                    max_dist_sq = d
                    pivot1, pivot2 = i, j

        # Split into two groups based on proximity to pivots
        parts_1 = num_parts // 2
        parts_2 = num_parts - parts_1
        target_rev_1 = target_rev_per_part * parts_1
        target_rev_2 = target_rev_per_part * parts_2

        group1 = [pivot1]
        group2 = [pivot2]
        rev1 = revenues[pivot1]
        rev2 = revenues[pivot2]

        remaining = [idx for idx in account_indices if idx != pivot1 and idx != pivot2]

        # Sort remaining by distance difference to pivots
        dist_sq_p1 = dist_sq[pivot1]
        dist_sq_p2 = dist_sq[pivot2]
        remaining.sort(key=lambda idx: dist_sq_p1[idx] - dist_sq_p2[idx])

        for idx in remaining:
            d1 = dist_sq_p1[idx]
            d2 = dist_sq_p2[idx]
            rev_idx = revenues[idx]

            # Score based on distance and revenue balance
            score1 = d1 + abs(rev1 + rev_idx - target_rev_1) * 0.00001
            score2 = d2 + abs(rev2 + rev_idx - target_rev_2) * 0.00001

            # Bias towards balance if one group is overfilled
            if rev1 > target_rev_1 * 1.2:
                score1 *= 2
            if rev2 > target_rev_2 * 1.2:
                score2 *= 2

            if score1 <= score2:
                group1.append(idx)
                rev1 += rev_idx
            else:
                group2.append(idx)
                rev2 += rev_idx

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

            # Compactness bonus: prefer moving to nearby accounts in min_part
            if partitions[min_part]:
                dist_sq_idx = dist_sq[idx]
                min_dist_to_target = min(dist_sq_idx[other] for other in partitions[min_part])
                compactness_bonus = 1.0 / (1.0 + min_dist_to_target)
            else:
                compactness_bonus = 0

            # Combined score favoring both balance and compactness
            combined = improvement + compactness_bonus * 0.01

            if combined > best_improvement:
                best_improvement = combined
                best_move = idx

        if best_move is not None and best_improvement > 0:
            partitions[max_part].remove(best_move)
            partitions[min_part].append(best_move)

    # Convert to output format
    result = {}
    for rep_id, partition in enumerate(partitions):
        result[rep_id] = [ids[idx] for idx in partition]

    return result
