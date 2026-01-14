"""
Variant gen6x: Hybrid crossover combining best features from gen3x, gen3b, gen1b.

Parents:
- gen3x (0.8168): Precomputed distances, K-means++ seeding, sorted assignment with
                  incremental centroids, revenue-aware scoring, compactness bonus
- gen3b (0.8097): Multi-pair balance optimization (ALL over/under pairs), k-means iteration
- gen1b (0.7993): Precomputed sorted neighbor lists, efficient pivot/neighbor lookups

Crossover design:
1. From gen3x: Precomputed squared distance matrix for O(1) lookups
2. From gen3x: K-means++ style seed initialization (center of mass + max-min spread)
3. From gen3x: Sorted assignment order (closest first) with incremental centroid updates
4. From gen3x: Revenue-aware scoring with overfill penalty during assignment
5. From gen3b: Multi-pair balance optimization - try ALL over->under territory pairs
6. From gen3b: Double k-means iteration before balancing for tighter initial clustering
7. From gen1b: Precomputed sorted neighbor lists for efficient compactness calculations
8. Combined: Enhanced compactness scoring using neighbor lists during balancing

Hypothesis: Combining the best initial assignment from gen3x with gen3b's exhaustive
multi-pair exploration and gen1b's efficient neighbor lookups should yield better
overall balance while maintaining compactness.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid clustering with multi-pair balance and precomputed neighbor lookups."""
    if num_reps <= 0:
        return {}
    if not accounts:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data (common to all parents)
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # From gen3x + gen1b: Precompute all pairwise squared distances for O(1) lookups
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2
            dist_sq[i][j] = d_sq
            dist_sq[j][i] = d_sq

    # From gen1b: Precompute sorted neighbor lists for efficient compactness lookups
    # Only compute for use in balancing phase (top-k nearest neighbors)
    k_neighbors = min(20, n - 1)  # Limit to 20 nearest for efficiency
    nearest_neighbors = []
    for i in range(n):
        neighbor_list = [(j, dist_sq[i][j]) for j in range(n) if j != i]
        neighbor_list.sort(key=lambda x: x[1])
        nearest_neighbors.append([x[0] for x in neighbor_list[:k_neighbors]])

    # From gen3x: K-means++ style seed initialization for maximum spread
    seeds = []
    used = set()

    # First seed: center of mass
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

    # From gen3x: Sort accounts by distance to nearest seed (closest first)
    def min_seed_dist(i):
        return min(dist_sq[i][s] for s in seeds)

    sorted_indices = sorted(range(n), key=min_seed_dist)

    # From gen3x: Assignment with incremental centroid updates and revenue-aware scoring
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0.0] * num_reps

    for i in sorted_indices:
        best_score = float("inf")
        best_rep = 0

        for rep_id in range(num_reps):
            # Distance to current centroid
            dist = (lats[i] - centroid_lats[rep_id]) ** 2 + (lons[i] - centroid_lons[rep_id]) ** 2

            # Revenue balance factor (from gen3x)
            rev_deviation = abs(rep_revenues[rep_id] + revenues[i] - target_revenue)
            rev_factor = rev_deviation * 0.00001

            # Penalty for overfilled territories (from gen3x)
            if rep_revenues[rep_id] > target_revenue * 1.2:
                dist *= 1.5

            score = dist + rev_factor

            if score < best_score:
                best_score = score
                best_rep = rep_id

        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += revenues[i]

        # From gen3x: Incremental centroid update
        k = len(rep_accounts[best_rep])
        centroid_lats[best_rep] = sum(lats[idx] for idx in rep_accounts[best_rep]) / k
        centroid_lons[best_rep] = sum(lons[idx] for idx in rep_accounts[best_rep]) / k

    # From gen3b: Extra k-means iteration - reassign with updated centroids
    # Reset and reassign
    for rep_id in range(num_reps):
        rep_accounts[rep_id] = []
        rep_revenues[rep_id] = 0.0

    for i in range(n):
        min_dist = float("inf")
        best_rep = 0
        for rep_id in range(num_reps):
            d = (lats[i] - centroid_lats[rep_id]) ** 2 + (lons[i] - centroid_lons[rep_id]) ** 2
            if d < min_dist:
                min_dist = d
                best_rep = rep_id
        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += revenues[i]

    # Update centroids after reassignment
    for rep_id in range(num_reps):
        if rep_accounts[rep_id]:
            centroid_lats[rep_id] = sum(lats[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])
            centroid_lons[rep_id] = sum(lons[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])

    # From gen3b + gen3x: Multi-pair balance optimization with compactness bonus
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    # Create account to rep mapping for efficient lookups during balancing
    account_to_rep = {i: assignments[i] for i in range(n)}

    for iteration in range(150):  # From gen3b: more iterations
        current_cv = compute_cv()
        if current_cv < 0.02:  # Tight CV threshold
            break

        # From gen3b: Find ALL over-target and under-target territories
        over_target = [r for r in range(num_reps) if rep_revenues[r] > target_revenue]
        under_target = [r for r in range(num_reps) if rep_revenues[r] < target_revenue]

        if not over_target or not under_target:
            break

        best_improvement = 0
        best_action = None

        # From gen3b: Try moves/swaps between ALL over->under pairs
        for from_rep in over_target:
            for to_rep in under_target:
                # Try moving accounts from from_rep to to_rep
                for i in rep_accounts[from_rep]:
                    if len(rep_accounts[from_rep]) <= 1:
                        continue

                    new_revs = rep_revenues[:]
                    new_revs[from_rep] -= revenues[i]
                    new_revs[to_rep] += revenues[i]

                    new_mean = sum(new_revs) / num_reps
                    new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                    new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                    improvement = current_cv - new_cv

                    # From gen1b + gen3x: Compactness bonus using nearest neighbors
                    if rep_accounts[to_rep]:
                        # Check if any of i's nearest neighbors are in to_rep
                        neighbor_bonus = 0
                        for neighbor in nearest_neighbors[i]:
                            if account_to_rep.get(neighbor) == to_rep:
                                neighbor_bonus = 0.001 / (1.0 + dist_sq[i][neighbor])
                                break  # Use first (nearest) neighbor found
                        improvement += neighbor_bonus

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_action = ("move", i, from_rep, to_rep)

                # Try swapping accounts between from_rep and to_rep
                for i in rep_accounts[from_rep]:
                    for j in rep_accounts[to_rep]:
                        new_revs = rep_revenues[:]
                        new_revs[from_rep] = new_revs[from_rep] - revenues[i] + revenues[j]
                        new_revs[to_rep] = new_revs[to_rep] - revenues[j] + revenues[i]

                        new_mean = sum(new_revs) / num_reps
                        new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                        new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                        improvement = current_cv - new_cv
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_action = ("swap", i, j, from_rep, to_rep)

        if best_action is None or best_improvement < 0.0005:
            break

        # Apply the best action
        if best_action[0] == "move":
            _, i, from_rep, to_rep = best_action
            assignments[i] = to_rep
            account_to_rep[i] = to_rep
            rep_accounts[from_rep].remove(i)
            rep_accounts[to_rep].append(i)
            rep_revenues[from_rep] -= revenues[i]
            rep_revenues[to_rep] += revenues[i]
        else:  # swap
            _, i, j, rep_i, rep_j = best_action
            assignments[i] = rep_j
            assignments[j] = rep_i
            account_to_rep[i] = rep_j
            account_to_rep[j] = rep_i
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
