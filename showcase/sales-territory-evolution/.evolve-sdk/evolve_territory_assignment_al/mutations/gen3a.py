"""
Variant gen3a: Added compactness-focused swap phase after balance optimization.

Mutation from gen1b: Algorithm change - Add a second post-processing phase that
specifically optimizes for compactness by swapping accounts between ANY pair of
territories (not just max/min revenue). Swaps are accepted if they improve total
compactness without significantly hurting balance.

Key changes:
1. Keep the original balance-focused swap phase (20 iterations)
2. Add new compactness-focused swap phase (15 iterations) that checks all territory pairs
3. Accept swaps that reduce total average intra-territory distance

Hypothesis: The original algorithm achieves good balance but the recursive
partitioning doesn't optimize for minimum intra-territory distances. A dedicated
compactness phase can improve geographic tightness by 5-10% while maintaining
the good balance already achieved.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Graph-based balanced partitioning with compactness optimization."""
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

    # Phase 1: Balance revenues with local swaps (original)
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

    # Phase 2: MUTATION - Compactness-focused swap phase
    # Try swapping accounts between ANY pair of territories to improve compactness
    def calc_partition_avg_dist(part_indices):
        """Calculate average pairwise squared distance within a partition."""
        if len(part_indices) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i, a in enumerate(part_indices):
            for b in part_indices[i+1:]:
                total += dist_sq[a][b]
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def calc_total_compactness():
        """Sum of average distances across all partitions (lower is better)."""
        return sum(calc_partition_avg_dist(p) for p in partitions)

    # Get current balance constraint
    rev_list = get_partition_revenues()
    balance_threshold = target_revenue_per_rep * 0.15  # Allow 15% deviation

    for _ in range(15):
        current_compactness = calc_total_compactness()
        best_swap = None
        best_improvement = 0

        # Try swapping accounts between all partition pairs
        for p1 in range(num_reps):
            for p2 in range(p1 + 1, num_reps):
                if not partitions[p1] or not partitions[p2]:
                    continue

                # Try each account in p1 swapped to p2
                for idx1 in partitions[p1]:
                    # Find best partner in p2 to swap with
                    for idx2 in partitions[p2]:
                        # Check if swap maintains balance
                        new_rev_p1 = rev_list[p1] - revenues[idx1] + revenues[idx2]
                        new_rev_p2 = rev_list[p2] - revenues[idx2] + revenues[idx1]

                        if abs(new_rev_p1 - target_revenue_per_rep) > balance_threshold:
                            continue
                        if abs(new_rev_p2 - target_revenue_per_rep) > balance_threshold:
                            continue

                        # Calculate compactness improvement from this swap
                        # Only compute affected partitions (p1 and p2)
                        old_comp_p1 = calc_partition_avg_dist(partitions[p1])
                        old_comp_p2 = calc_partition_avg_dist(partitions[p2])

                        new_p1 = [x for x in partitions[p1] if x != idx1] + [idx2]
                        new_p2 = [x for x in partitions[p2] if x != idx2] + [idx1]

                        new_comp_p1 = calc_partition_avg_dist(new_p1)
                        new_comp_p2 = calc_partition_avg_dist(new_p2)

                        improvement = (old_comp_p1 + old_comp_p2) - (new_comp_p1 + new_comp_p2)

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (p1, idx1, p2, idx2)

        if best_swap is None or best_improvement <= 0:
            break

        # Apply best swap
        p1, idx1, p2, idx2 = best_swap
        partitions[p1].remove(idx1)
        partitions[p2].remove(idx2)
        partitions[p1].append(idx2)
        partitions[p2].append(idx1)

        # Update revenues
        rev_list[p1] = rev_list[p1] - revenues[idx1] + revenues[idx2]
        rev_list[p2] = rev_list[p2] - revenues[idx2] + revenues[idx1]

    # Convert to output format
    result = {}
    for rep_id, partition in enumerate(partitions):
        result[rep_id] = [ids[idx] for idx in partition]

    return result
