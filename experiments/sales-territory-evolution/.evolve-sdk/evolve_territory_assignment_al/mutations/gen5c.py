"""
Variant gen5c: Optimized pivot selection with greedy farthest-pair heuristic.

Parent: gen1b (0.7993)

Mutation: Branch elimination + Loop optimization
- Replace the slow O(n*m) pivot search (iterating reversed neighbors looking for
  membership in idx_set) with a faster greedy farthest-pair heuristic
- Use sampling when subset is large to reduce pivot search overhead
- Eliminate expensive set membership lookups in inner loops

Key changes:
1. For small subsets (<=20): use direct O(n^2) max-distance calculation (faster
   than repeated set lookups)
2. For larger subsets: use sampling heuristic - pick farthest point from first
   element, then farthest from that
3. Increased balancing iterations (20 -> 30) with early termination on plateau

Hypothesis: The original pivot search has poor cache locality due to random
access patterns in reversed(neighbors[i]). Direct computation for small subsets
and sampling for large subsets eliminates branch mispredictions and improves
cache efficiency, leading to better overall performance.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with optimized pivot selection."""
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

    def find_farthest_pair_direct(indices):
        """Direct O(n^2) search for farthest pair - best for small subsets."""
        max_d = -1
        p1, p2 = indices[0], indices[1] if len(indices) > 1 else indices[0]
        for i_idx, i in enumerate(indices):
            for j in indices[i_idx + 1:]:
                d = dist_sq[i][j]
                if d > max_d:
                    max_d = d
                    p1, p2 = i, j
        return p1, p2

    def find_farthest_pair_greedy(indices):
        """Greedy 2-step heuristic - good for larger subsets."""
        # Start from first element
        start = indices[0]

        # Find farthest point from start
        max_d = -1
        far1 = start
        for j in indices:
            if dist_sq[start][j] > max_d:
                max_d = dist_sq[start][j]
                far1 = j

        # Find farthest point from far1
        max_d = -1
        far2 = far1
        for j in indices:
            if dist_sq[far1][j] > max_d:
                max_d = dist_sq[far1][j]
                far2 = j

        return far1, far2

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

        # MUTATION: Optimized pivot selection
        # Use direct search for small subsets, greedy heuristic for larger ones
        if len(account_indices) <= 20:
            pivot1, pivot2 = find_farthest_pair_direct(account_indices)
        else:
            pivot1, pivot2 = find_farthest_pair_greedy(account_indices)

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

    prev_imbalance = float('inf')
    for _ in range(30):  # Increased from 20 to 30
        rev_list = get_partition_revenues()
        max_part = max(range(num_reps), key=lambda p: rev_list[p])
        min_part = min(range(num_reps), key=lambda p: rev_list[p])

        if max_part == min_part:
            break

        imbalance = rev_list[max_part] - rev_list[min_part]
        if imbalance < target_revenue_per_rep * 0.1:
            break  # Good enough balance

        # Early termination on plateau
        if abs(prev_imbalance - imbalance) < target_revenue_per_rep * 0.001:
            break
        prev_imbalance = imbalance

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
                min_dist_to_target = min(dist_sq[idx][other] for other in partitions[min_part])
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
