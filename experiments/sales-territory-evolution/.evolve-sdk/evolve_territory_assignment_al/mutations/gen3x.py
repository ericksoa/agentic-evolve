"""
Variant gen3x: Hybrid crossover combining best features from gen1a, gen1b, gen1c.

Parents:
- gen1b (0.7993): Precomputed distance matrix, recursive partitioning with pivots
- gen1c (0.7919): K-means++ initialization, aggressive CV-based balancing
- gen1a (0.7890): Incremental centroid updates, sorted assignment order

Crossover design:
1. From gen1b: Precomputed squared distance matrix for O(1) lookups
2. From gen1b: Revenue-aware scoring during assignment phase
3. From gen1c: K-means++ style seed initialization for maximum spread
4. From gen1c: Aggressive balancing with both moves and swaps, CV termination
5. From gen1a: Sorted assignment order (closest first) with incremental updates

Hypothesis: Combining precomputed distances with k-means++ seeding and incremental
updates should yield better compactness, while the aggressive CV-based balancing
ensures revenue balance.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid clustering with precomputed distances and aggressive balancing."""
    if num_reps <= 0:
        return {}
    if not accounts:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data (from all parents)
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # From gen1b: Precompute all pairwise squared distances for O(1) lookups
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2
            dist_sq[i][j] = d_sq
            dist_sq[j][i] = d_sq

    # From gen1c: K-means++ style seed initialization for maximum spread
    seeds = []  # Store seed indices
    used = set()

    # First seed: center of mass (from gen1c)
    avg_lat = sum(lats) / n
    avg_lon = sum(lons) / n

    # Find account closest to center of mass
    min_dist = float("inf")
    first_idx = 0
    for i in range(n):
        d = (lats[i] - avg_lat) ** 2 + (lons[i] - avg_lon) ** 2
        if d < min_dist:
            min_dist = d
            first_idx = i

    seeds.append(first_idx)
    used.add(first_idx)

    # Remaining seeds: max-min distance selection (k-means++)
    for _ in range(1, num_reps):
        max_min_dist = -1
        best_idx = 0
        for i in range(n):
            if i in used:
                continue
            # Use precomputed distances
            min_d = min(dist_sq[i][s] for s in seeds)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_idx = i
        used.add(best_idx)
        seeds.append(best_idx)

    # Initialize centroids at seed locations
    centroid_lats = [lats[s] for s in seeds]
    centroid_lons = [lons[s] for s in seeds]

    # From gen1a: Sort accounts by distance to nearest seed (closest first)
    def min_seed_dist(i):
        return min(dist_sq[i][s] for s in seeds)

    sorted_indices = sorted(range(n), key=min_seed_dist)

    # From gen1a + gen1b: Assignment with incremental centroid updates
    # and revenue-aware scoring
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0.0] * num_reps

    for i in sorted_indices:
        # Find best territory considering distance and revenue balance (from gen1b)
        best_score = float("inf")
        best_rep = 0

        for rep_id in range(num_reps):
            # Distance to current centroid
            dist = (lats[i] - centroid_lats[rep_id]) ** 2 + (lons[i] - centroid_lons[rep_id]) ** 2

            # Revenue balance factor (from gen1b: score with revenue bias)
            rev_deviation = abs(rep_revenues[rep_id] + revenues[i] - target_revenue)
            rev_factor = rev_deviation * 0.00001  # Small weight to break ties

            # Penalty for overfilled territories (from gen1b)
            if rep_revenues[rep_id] > target_revenue * 1.2:
                dist *= 1.5

            score = dist + rev_factor

            if score < best_score:
                best_score = score
                best_rep = rep_id

        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += revenues[i]

        # From gen1a: Incremental centroid update
        k = len(rep_accounts[best_rep])
        centroid_lats[best_rep] = sum(lats[idx] for idx in rep_accounts[best_rep]) / k
        centroid_lons[best_rep] = sum(lons[idx] for idx in rep_accounts[best_rep]) / k

    # From gen1c: Aggressive revenue balancing with CV-based termination
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    # From gen1b + gen1c: Balancing with compactness consideration
    for iteration in range(100):
        current_cv = compute_cv()
        if current_cv < 0.02:  # Tighter than gen1c (5%), looser than gen1a (1%)
            break

        # Find most overloaded and most underloaded reps
        max_rep = max(range(num_reps), key=lambda r: rep_revenues[r])
        min_rep = min(range(num_reps), key=lambda r: rep_revenues[r])

        if max_rep == min_rep:
            break

        best_improvement = 0
        best_action = None

        # Try moving accounts from max_rep to min_rep (from gen1c)
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

            # From gen1b: Compactness bonus for moving to nearby accounts
            if rep_accounts[min_rep]:
                min_dist_to_target = min(dist_sq[i][other] for other in rep_accounts[min_rep])
                compactness_bonus = 0.001 / (1.0 + min_dist_to_target)
                improvement += compactness_bonus

            if improvement > best_improvement:
                best_improvement = improvement
                best_action = ("move", i, max_rep, min_rep)

        # Try swapping accounts between max_rep and min_rep (from gen1c)
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
