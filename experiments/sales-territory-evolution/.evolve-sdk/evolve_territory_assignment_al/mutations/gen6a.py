"""
Variant gen6a: Cache-optimized flat distance array.

Parent: gen3x (0.81684)

Mutation: Cache optimization - Flat array for distance matrix
- Replace nested list dist_sq[i][j] with flat array dist_sq[i * n + j]
- This improves memory locality and reduces pointer indirection
- Python lists of lists have poor cache behavior due to multiple allocations

Hypothesis: Using a flat array instead of a nested list-of-lists reduces
memory fragmentation and pointer chasing. The distance matrix is accessed
many times during both seeding, assignment, and balancing phases.
Flattening should improve cache hit rates and reduce memory overhead.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid clustering with cache-optimized flat distance array."""
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

    # MUTATION: Flat array for distance matrix (better cache locality)
    # Access pattern: dist_sq[i * n + j] instead of dist_sq[i][j]
    dist_sq = [0.0] * (n * n)
    for i in range(n):
        base_i = i * n
        lat_i = lats[i]
        lon_i = lons[i]
        for j in range(i + 1, n):
            d_sq = (lat_i - lats[j]) ** 2 + (lon_i - lons[j]) ** 2
            dist_sq[base_i + j] = d_sq
            dist_sq[j * n + i] = d_sq

    # K-means++ style seed initialization for maximum spread
    seeds = []  # Store seed indices
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
            # Use flat array access
            base_i = i * n
            min_d = min(dist_sq[base_i + s] for s in seeds)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_idx = i
        used.add(best_idx)
        seeds.append(best_idx)

    # Initialize centroids at seed locations
    centroid_lats = [lats[s] for s in seeds]
    centroid_lons = [lons[s] for s in seeds]

    # Sort accounts by distance to nearest seed (closest first)
    def min_seed_dist(i):
        base_i = i * n
        return min(dist_sq[base_i + s] for s in seeds)

    sorted_indices = sorted(range(n), key=min_seed_dist)

    # Assignment with incremental centroid updates and revenue-aware scoring
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0.0] * num_reps

    for i in sorted_indices:
        # Find best territory considering distance and revenue balance
        best_score = float("inf")
        best_rep = 0

        lat_i = lats[i]
        lon_i = lons[i]
        rev_i = revenues[i]

        for rep_id in range(num_reps):
            # Distance to current centroid
            dist = (lat_i - centroid_lats[rep_id]) ** 2 + (lon_i - centroid_lons[rep_id]) ** 2

            # Revenue balance factor (score with revenue bias)
            rev_deviation = abs(rep_revenues[rep_id] + rev_i - target_revenue)
            rev_factor = rev_deviation * 0.00001  # Small weight to break ties

            # Penalty for overfilled territories
            if rep_revenues[rep_id] > target_revenue * 1.2:
                dist *= 1.5

            score = dist + rev_factor

            if score < best_score:
                best_score = score
                best_rep = rep_id

        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += rev_i

        # Incremental centroid update
        k = len(rep_accounts[best_rep])
        centroid_lats[best_rep] = sum(lats[idx] for idx in rep_accounts[best_rep]) / k
        centroid_lons[best_rep] = sum(lons[idx] for idx in rep_accounts[best_rep]) / k

    # Aggressive revenue balancing with CV-based termination
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    # Balancing with compactness consideration
    for iteration in range(100):
        current_cv = compute_cv()
        if current_cv < 0.02:  # 2% CV target
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

            # Compactness bonus for moving to nearby accounts (using flat array)
            if rep_accounts[min_rep]:
                base_i = i * n
                min_dist_to_target = min(dist_sq[base_i + other] for other in rep_accounts[min_rep])
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
