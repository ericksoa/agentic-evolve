"""
Variant gen2a: Eliminate sorted neighbor lists with direct farthest pair search.

Mutation from gen1b: Cache optimization - Remove the O(n² log n) neighbor list
precomputation and replace with direct O(k²) farthest pair search within each
partition subset, where k is the subset size. This trades precomputation for
localized computation that's more cache-friendly.

Key changes:
1. Remove sorted neighbor lists and farthest array
2. Use direct iteration for farthest pair in each partition
3. Use array-based membership check instead of set for small subsets
4. Optimize swap phase with early exit conditions

Hypothesis: For typical partition sizes (n/num_reps accounts per partition),
direct O(k²) search within the subset is faster than maintaining O(n) neighbor
lists plus O(n) set lookups. Better cache locality on smaller working sets.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with optimized local search."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data into contiguous arrays for cache efficiency
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)

    # Precompute all pairwise squared distances (avoid sqrt during lookup)
    # Use flat array for better cache performance
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        lat_i = lats[i]
        lon_i = lons[i]
        row = dist_sq[i]
        for j in range(i + 1, n):
            d_sq = (lat_i - lats[j])**2 + (lon_i - lons[j])**2
            row[j] = d_sq
            dist_sq[j][i] = d_sq

    def partition_accounts(account_indices, num_parts, target_rev_per_part):
        """Recursively partition accounts into num_parts balanced groups."""
        if num_parts == 1:
            return [account_indices]

        k = len(account_indices)
        if k <= num_parts:
            # Not enough accounts - assign one per part
            result = [[idx] for idx in account_indices]
            while len(result) < num_parts:
                result.append([])
            return result

        # Find two accounts that are farthest apart (pivots)
        # Direct O(k²) search - better for small k than O(n) neighbor lookup with set
        max_dist_sq = -1
        pivot1, pivot2 = account_indices[0], account_indices[1]

        if k <= 50:
            # For small subsets, direct pairwise is cache-friendly
            for ii in range(k):
                i = account_indices[ii]
                row_i = dist_sq[i]
                for jj in range(ii + 1, k):
                    j = account_indices[jj]
                    d = row_i[j]
                    if d > max_dist_sq:
                        max_dist_sq = d
                        pivot1, pivot2 = i, j
        else:
            # For larger subsets, sample to find approximate farthest pair
            # Check every 3rd element to reduce comparisons
            step = 3
            for ii in range(0, k, step):
                i = account_indices[ii]
                row_i = dist_sq[i]
                for jj in range(ii + step, k, step):
                    j = account_indices[jj]
                    d = row_i[j]
                    if d > max_dist_sq:
                        max_dist_sq = d
                        pivot1, pivot2 = i, j

            # Refine: find true farthest from each sampled pivot
            for candidate in (pivot1, pivot2):
                row = dist_sq[candidate]
                for jj in range(k):
                    j = account_indices[jj]
                    if j != candidate:
                        d = row[j]
                        if d > max_dist_sq:
                            max_dist_sq = d
                            pivot1, pivot2 = candidate, j

        # Split into two groups based on proximity to pivots
        parts_1 = num_parts // 2
        parts_2 = num_parts - parts_1
        target_rev_1 = target_rev_per_part * parts_1
        target_rev_2 = target_rev_per_part * parts_2

        group1 = [pivot1]
        group2 = [pivot2]
        rev1 = revenues[pivot1]
        rev2 = revenues[pivot2]

        # Build remaining list and sort by distance difference
        remaining = []
        pivot1_row = dist_sq[pivot1]
        pivot2_row = dist_sq[pivot2]
        for idx in account_indices:
            if idx != pivot1 and idx != pivot2:
                diff = pivot1_row[idx] - pivot2_row[idx]
                remaining.append((diff, idx))

        remaining.sort()

        rev_weight = 0.00001
        thresh_1 = target_rev_1 * 1.2
        thresh_2 = target_rev_2 * 1.2

        for diff, idx in remaining:
            d1 = pivot1_row[idx]
            d2 = pivot2_row[idx]
            rev_idx = revenues[idx]

            # Score based on distance and revenue balance
            score1 = d1 + abs(rev1 + rev_idx - target_rev_1) * rev_weight
            score2 = d2 + abs(rev2 + rev_idx - target_rev_2) * rev_weight

            # Bias towards balance if one group is overfilled
            if rev1 > thresh_1:
                score1 *= 2
            if rev2 > thresh_2:
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
    # Enhanced: use early exit and avoid redundant calculations
    target = target_revenue_per_rep
    thresh_good = target * 0.1

    for iteration in range(20):
        # Calculate revenues once per iteration
        rev_list = [sum(revenues[idx] for idx in part) for part in partitions]

        max_rev = -1
        min_rev = float('inf')
        max_part = 0
        min_part = 0

        for p in range(num_reps):
            r = rev_list[p]
            if r > max_rev:
                max_rev = r
                max_part = p
            if r < min_rev:
                min_rev = r
                min_part = p

        if max_part == min_part:
            break

        imbalance = max_rev - min_rev
        if imbalance < thresh_good:
            break  # Good enough balance

        # Try to find an account to move
        best_move = None
        best_improvement = 0

        max_part_list = partitions[max_part]
        min_part_list = partitions[min_part]
        old_imbalance = abs(max_rev - target) + abs(min_rev - target)

        for idx in max_part_list:
            rev_idx = revenues[idx]
            new_imbalance = abs(max_rev - rev_idx - target) + abs(min_rev + rev_idx - target)
            improvement = old_imbalance - new_imbalance

            # Compactness bonus: prefer moving to nearby accounts in min_part
            if min_part_list:
                row_idx = dist_sq[idx]
                min_dist_to_target = min(row_idx[other] for other in min_part_list)
                compactness_bonus = 1.0 / (1.0 + min_dist_to_target)
            else:
                compactness_bonus = 0

            # Combined score favoring both balance and compactness
            combined = improvement + compactness_bonus * 0.01

            if combined > best_improvement:
                best_improvement = combined
                best_move = idx

        if best_move is not None and best_improvement > 0:
            max_part_list.remove(best_move)
            min_part_list.append(best_move)

    # Convert to output format
    result = {}
    for rep_id, partition in enumerate(partitions):
        result[rep_id] = [ids[idx] for idx in partition]

    return result
