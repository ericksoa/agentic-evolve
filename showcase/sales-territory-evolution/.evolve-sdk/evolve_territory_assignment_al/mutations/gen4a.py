"""
Variant gen4a: Restore recursive pivot partitioning for better compactness.

Mutation from gen3x: Algorithm swap - Replace k-means++ centroid-based initial
assignment with recursive pivot-based partitioning from gen1b.

Key insight: gen3x's compactness (0.7113) dropped significantly from gen1b (0.8139)
because it replaced the recursive pivot partitioning with centroid-based assignment.
The pivot approach creates more geographically coherent initial clusters.

Key changes:
1. Use recursive pivot-based partitioning (from gen1b) for initial assignment
2. Keep gen3x's aggressive CV-based balancing with moves and swaps
3. Retain precomputed distance matrix and compactness bonus

Hypothesis: Recursive partitioning creates better initial clusters by explicitly
maximizing geographic separation at each split, leading to better compactness
while the aggressive balancing phase maintains revenue balance.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Recursive pivot partitioning with aggressive CV-based balancing."""
    if num_reps <= 0:
        return {}
    if not accounts:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # Precompute all pairwise squared distances for O(1) lookups
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2
            dist_sq[i][j] = d_sq
            dist_sq[j][i] = d_sq

    # Precompute sorted neighbor lists (nearest to farthest) for pivot selection
    neighbors = []
    for i in range(n):
        neighbor_list = [(j, dist_sq[i][j]) for j in range(n) if j != i]
        neighbor_list.sort(key=lambda x: x[1])
        neighbors.append([x[0] for x in neighbor_list])

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
        max_dist_sq = -1
        pivot1, pivot2 = account_indices[0], account_indices[1] if len(account_indices) > 1 else account_indices[0]

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
                score1 *= 1.5
            if rev2 > target_rev_2 * 1.2:
                score2 *= 1.5

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

    # Initial partitioning using recursive pivot-based approach
    all_indices = list(range(n))
    partitions = partition_accounts(all_indices, num_reps, target_revenue)

    # Convert partitions to assignment structures for balancing phase
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0.0] * num_reps

    for rep_id, partition in enumerate(partitions):
        for idx in partition:
            assignments[idx] = rep_id
            rep_accounts[rep_id].append(idx)
            rep_revenues[rep_id] += revenues[idx]

    # From gen3x: Aggressive revenue balancing with CV-based termination
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    # Balancing with compactness consideration (from gen3x)
    for iteration in range(100):
        current_cv = compute_cv()
        if current_cv < 0.02:
            break

        # Find most overloaded and most underloaded reps
        max_rep = max(range(num_reps), key=lambda r: rep_revenues[r])
        min_rep = min(range(num_reps), key=lambda r: rep_revenues[r])

        if max_rep == min_rep:
            break

        best_improvement = 0
        best_action = None

        # Try moving accounts from max_rep to min_rep
        for i in rep_accounts[max_rep]:
            if len(rep_accounts[max_rep]) <= 1:
                continue  # Don't empty a territory

            new_revs = rep_revenues[:]
            new_revs[max_rep] -= revenues[i]
            new_revs[min_rep] += revenues[i]

            new_mean = sum(new_revs) / num_reps
            new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
            new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

            improvement = current_cv - new_cv

            # Compactness bonus for moving to nearby accounts
            if rep_accounts[min_rep]:
                min_dist_to_target = min(dist_sq[i][other] for other in rep_accounts[min_rep])
                compactness_bonus = 0.001 / (1.0 + min_dist_to_target)
                improvement += compactness_bonus

            if improvement > best_improvement:
                best_improvement = improvement
                best_action = ("move", i, max_rep, min_rep)

        # Try swapping accounts between max_rep and min_rep
        for i in rep_accounts[max_rep]:
            for j in rep_accounts[min_rep]:
                new_revs = rep_revenues[:]
                new_revs[max_rep] = new_revs[max_rep] - revenues[i] + revenues[j]
                new_revs[min_rep] = new_revs[min_rep] - revenues[j] + revenues[i]

                new_mean = sum(new_revs) / num_reps
                new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                improvement = current_cv - new_cv

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = ("swap", i, j, max_rep, min_rep)

        if best_action is None or best_improvement < 0.0001:
            break

        # Apply the best action
        if best_action[0] == "move":
            _, i, from_rep, to_rep = best_action
            assignments[i] = to_rep
            rep_accounts[from_rep].remove(i)
            rep_accounts[to_rep].append(i)
            rep_revenues[from_rep] -= revenues[i]
            rep_revenues[to_rep] += revenues[i]
        else:  # swap
            _, i, j, rep_i, rep_j = best_action
            assignments[i] = rep_j
            assignments[j] = rep_i
            rep_accounts[rep_i].remove(i)
            rep_accounts[rep_i].append(j)
            rep_accounts[rep_j].remove(j)
            rep_accounts[rep_j].append(i)
            rep_revenues[rep_i] = rep_revenues[rep_i] - revenues[i] + revenues[j]
            rep_revenues[rep_j] = rep_revenues[rep_j] - revenues[j] + revenues[i]

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
