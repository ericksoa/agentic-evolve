"""
Variant gen7b: Early-exit greedy balancing with "good enough" threshold.

Parent: gen3b (fitness 0.8096)

Mutation type: Hybrid dispatch / Branch elimination
- Instead of exhaustively evaluating ALL potential moves/swaps to find the BEST one,
  use an early-exit strategy: accept the FIRST move that improves CV by at least a threshold
- Only fall back to exhaustive search when no "good enough" move is found quickly
- This dramatically reduces the O(k²n²) complexity per iteration to O(kn) in most cases

Key changes:
1. Add "good_enough_threshold" - accept first move that improves CV by at least this amount
2. Search in greedy order (most imbalanced pairs first)
3. Only continue exhaustive search if greedy pass finds nothing

Hypothesis: The parent exhaustively evaluates ALL move/swap combinations per iteration,
which is expensive for large inputs. In practice, many moves provide similar improvement.
By accepting the first "good enough" move, we can complete more iterations faster
while still making progress toward balance. This trades optimal per-step choices
for faster convergence overall.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Greedy centroid clustering with early-exit balance optimization."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # K-means++ style seed initialization for maximum spread
    centroids_lat = []
    centroids_lon = []
    used = set()

    # First centroid: center of mass
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

    centroids_lat.append(lats[first_idx])
    centroids_lon.append(lons[first_idx])
    used.add(first_idx)

    # Remaining centroids: max-min distance selection (k-means++)
    for _ in range(1, num_reps):
        max_min_dist = -1
        best_idx = 0
        for i in range(n):
            if i in used:
                continue
            # Find min distance to existing centroids
            min_d = min(
                (lats[i] - clat) ** 2 + (lons[i] - clon) ** 2
                for clat, clon in zip(centroids_lat, centroids_lon)
            )
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_idx = i
        used.add(best_idx)
        centroids_lat.append(lats[best_idx])
        centroids_lon.append(lons[best_idx])

    # Greedy assignment: each account to nearest centroid
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0] * num_reps

    for i in range(n):
        min_dist = float("inf")
        best_rep = 0
        for rep_id in range(num_reps):
            d = (lats[i] - centroids_lat[rep_id]) ** 2 + (lons[i] - centroids_lon[rep_id]) ** 2
            if d < min_dist:
                min_dist = d
                best_rep = rep_id
        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += revenues[i]

    # Update centroids based on actual assignments
    for rep_id in range(num_reps):
        if rep_accounts[rep_id]:
            centroids_lat[rep_id] = sum(lats[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])
            centroids_lon[rep_id] = sum(lons[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])

    # Repeat assignment with updated centroids (single k-means iteration)
    for rep_id in range(num_reps):
        rep_accounts[rep_id] = []
        rep_revenues[rep_id] = 0

    for i in range(n):
        min_dist = float("inf")
        best_rep = 0
        for rep_id in range(num_reps):
            d = (lats[i] - centroids_lat[rep_id]) ** 2 + (lons[i] - centroids_lon[rep_id]) ** 2
            if d < min_dist:
                min_dist = d
                best_rep = rep_id
        assignments[i] = best_rep
        rep_accounts[best_rep].append(i)
        rep_revenues[best_rep] += revenues[i]

    # MUTATION: Early-exit greedy balance optimization
    # Accept "good enough" improvements immediately instead of exhaustive search

    # Precompute mean (constant since total revenue doesn't change)
    mean_rev = total_revenue / num_reps

    def compute_cv_fast():
        # Use precomputed mean
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev if mean_rev > 0 else 0

    def compute_new_cv(new_revs):
        # Use precomputed mean (it doesn't change during balancing)
        new_var = sum((r - mean_rev) ** 2 for r in new_revs) / num_reps
        return math.sqrt(new_var) / mean_rev if mean_rev > 0 else 0

    # Threshold for "good enough" improvement - accept first move that improves by at least this
    good_enough_threshold = 0.005  # 0.5% CV improvement is good enough to accept immediately

    for iteration in range(200):  # More iterations since each is faster
        current_cv = compute_cv_fast()
        if current_cv < 0.02:  # Target: 2% CV
            break

        # Find over-target and under-target territories, sorted by imbalance
        over_target = [(r, rep_revenues[r] - target_revenue)
                       for r in range(num_reps) if rep_revenues[r] > target_revenue]
        under_target = [(r, target_revenue - rep_revenues[r])
                        for r in range(num_reps) if rep_revenues[r] < target_revenue]

        if not over_target or not under_target:
            break

        # Sort by imbalance magnitude (most imbalanced first) for faster convergence
        over_target.sort(key=lambda x: -x[1])
        under_target.sort(key=lambda x: -x[1])

        found_action = None

        # Greedy early-exit search: try pairs in order of imbalance magnitude
        for from_rep, _ in over_target:
            if found_action:
                break
            for to_rep, _ in under_target:
                if found_action:
                    break

                # Try moves first (simpler, often sufficient)
                for i in rep_accounts[from_rep]:
                    if len(rep_accounts[from_rep]) <= 1:
                        continue

                    new_revs = rep_revenues[:]
                    new_revs[from_rep] -= revenues[i]
                    new_revs[to_rep] += revenues[i]

                    new_cv = compute_new_cv(new_revs)
                    improvement = current_cv - new_cv

                    # Early exit: accept first "good enough" move
                    if improvement >= good_enough_threshold:
                        found_action = ("move", i, from_rep, to_rep)
                        break

                # If no good move found, try swaps for this pair
                if not found_action:
                    for i in rep_accounts[from_rep]:
                        if found_action:
                            break
                        for j in rep_accounts[to_rep]:
                            new_revs = rep_revenues[:]
                            new_revs[from_rep] = new_revs[from_rep] - revenues[i] + revenues[j]
                            new_revs[to_rep] = new_revs[to_rep] - revenues[j] + revenues[i]

                            new_cv = compute_new_cv(new_revs)
                            improvement = current_cv - new_cv

                            if improvement >= good_enough_threshold:
                                found_action = ("swap", i, j, from_rep, to_rep)
                                break

        # If no "good enough" action found, fall back to finding best action
        if not found_action:
            best_improvement = 0
            for from_rep, _ in over_target:
                for to_rep, _ in under_target:
                    for i in rep_accounts[from_rep]:
                        if len(rep_accounts[from_rep]) <= 1:
                            continue

                        new_revs = rep_revenues[:]
                        new_revs[from_rep] -= revenues[i]
                        new_revs[to_rep] += revenues[i]

                        new_cv = compute_new_cv(new_revs)
                        improvement = current_cv - new_cv

                        if improvement > best_improvement:
                            best_improvement = improvement
                            found_action = ("move", i, from_rep, to_rep)

                    for i in rep_accounts[from_rep]:
                        for j in rep_accounts[to_rep]:
                            new_revs = rep_revenues[:]
                            new_revs[from_rep] = new_revs[from_rep] - revenues[i] + revenues[j]
                            new_revs[to_rep] = new_revs[to_rep] - revenues[j] + revenues[i]

                            new_cv = compute_new_cv(new_revs)
                            improvement = current_cv - new_cv

                            if improvement > best_improvement:
                                best_improvement = improvement
                                found_action = ("swap", i, j, from_rep, to_rep)

            if found_action is None or best_improvement < 0.0005:
                break

        # Apply the action
        if found_action[0] == "move":
            _, i, from_rep, to_rep = found_action
            assignments[i] = to_rep
            rep_accounts[from_rep].remove(i)
            rep_accounts[to_rep].append(i)
            rep_revenues[from_rep] -= revenues[i]
            rep_revenues[to_rep] += revenues[i]
        else:  # swap
            _, i, j, rep_i, rep_j = found_action
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
