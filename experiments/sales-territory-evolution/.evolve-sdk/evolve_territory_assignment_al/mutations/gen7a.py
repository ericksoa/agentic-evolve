"""
Variant gen7a: Compactness-focused post-processing phase.

Parent: gen3x (fitness 0.81684, compactness=0.7113)

Mutation type: Algorithm addition - Add a dedicated compactness optimization phase
that runs AFTER the revenue balancing phase. This phase performs targeted swaps
between adjacent territories to improve geographic cohesion while maintaining
revenue balance within acceptable bounds.

Key changes:
1. After CV-based balancing reaches target, add compactness optimization phase
2. Build territory adjacency graph based on geographic proximity
3. Try boundary swaps that improve compactness while keeping CV < 5%
4. Use convex hull area ratio as compactness proxy for swap decisions

Hypothesis: The parent achieves good balance (0.9806) but sacrifices compactness
(0.7113). By adding a dedicated compactness phase that only makes swaps that
maintain acceptable balance (CV < 5%), we can recover compactness without
sacrificing balance quality.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid clustering with precomputed distances, aggressive balancing, and compactness optimization."""
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

    # ========== NEW: Compactness optimization phase ==========
    # After balancing, optimize compactness through boundary swaps
    # that maintain acceptable revenue balance (CV < 5%)

    def compute_territory_spread(rep_id):
        """Compute the sum of squared distances from centroid for a territory."""
        if len(rep_accounts[rep_id]) <= 1:
            return 0.0

        # Compute centroid
        c_lat = sum(lats[idx] for idx in rep_accounts[rep_id]) / len(rep_accounts[rep_id])
        c_lon = sum(lons[idx] for idx in rep_accounts[rep_id]) / len(rep_accounts[rep_id])

        # Sum of squared distances to centroid (lower is more compact)
        return sum((lats[idx] - c_lat) ** 2 + (lons[idx] - c_lon) ** 2
                   for idx in rep_accounts[rep_id])

    def find_boundary_accounts(rep_id):
        """Find accounts that are closer to another territory's centroid."""
        boundary = []
        if len(rep_accounts[rep_id]) <= 1:
            return boundary

        # Current centroid
        c_lat = sum(lats[idx] for idx in rep_accounts[rep_id]) / len(rep_accounts[rep_id])
        c_lon = sum(lons[idx] for idx in rep_accounts[rep_id]) / len(rep_accounts[rep_id])

        for idx in rep_accounts[rep_id]:
            dist_to_own = (lats[idx] - c_lat) ** 2 + (lons[idx] - c_lon) ** 2

            # Check if closer to any other territory
            for other_rep in range(num_reps):
                if other_rep == rep_id or not rep_accounts[other_rep]:
                    continue
                other_c_lat = sum(lats[j] for j in rep_accounts[other_rep]) / len(rep_accounts[other_rep])
                other_c_lon = sum(lons[j] for j in rep_accounts[other_rep]) / len(rep_accounts[other_rep])
                dist_to_other = (lats[idx] - other_c_lat) ** 2 + (lons[idx] - other_c_lon) ** 2

                if dist_to_other < dist_to_own:
                    boundary.append((idx, other_rep, dist_to_own - dist_to_other))
                    break

        return boundary

    # Compactness optimization iterations
    max_cv_for_compactness = 0.05  # Allow CV up to 5% during compactness optimization

    for _ in range(50):  # Limited iterations for compactness
        best_compactness_improvement = 0
        best_swap = None

        # For each territory, find boundary accounts that could move
        for rep_id in range(num_reps):
            boundary = find_boundary_accounts(rep_id)

            for idx, target_rep, dist_improvement in boundary:
                if len(rep_accounts[rep_id]) <= 1:
                    continue

                # Check if move maintains acceptable revenue balance
                new_revs = rep_revenues[:]
                new_revs[rep_id] -= revenues[idx]
                new_revs[target_rep] += revenues[idx]

                new_mean = sum(new_revs) / num_reps
                new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                if new_cv > max_cv_for_compactness:
                    continue  # Skip moves that hurt balance too much

                # Compute compactness improvement
                old_spread = compute_territory_spread(rep_id) + compute_territory_spread(target_rep)

                # Simulate move
                rep_accounts[rep_id].remove(idx)
                rep_accounts[target_rep].append(idx)
                new_spread = compute_territory_spread(rep_id) + compute_territory_spread(target_rep)
                rep_accounts[target_rep].remove(idx)
                rep_accounts[rep_id].append(idx)

                compactness_improvement = old_spread - new_spread

                if compactness_improvement > best_compactness_improvement:
                    best_compactness_improvement = compactness_improvement
                    best_swap = (idx, rep_id, target_rep)

        if best_swap is None or best_compactness_improvement < 0.0001:
            break

        # Apply the compactness-improving move
        idx, from_rep, to_rep = best_swap
        assignments[idx] = to_rep
        rep_accounts[from_rep].remove(idx)
        rep_accounts[to_rep].append(idx)
        rep_revenues[from_rep] -= revenues[idx]
        rep_revenues[to_rep] += revenues[idx]

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
