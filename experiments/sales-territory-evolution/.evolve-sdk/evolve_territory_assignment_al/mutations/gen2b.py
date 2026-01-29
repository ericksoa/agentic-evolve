"""
Variant gen2b: Multiple k-means iterations for better convergence.

Mutation from gen1c: Loop unrolling - Replace single k-means iteration with
multiple iterations (5 iterations) to achieve better centroid convergence
before the balancing phase.

Key change:
- Run 5 k-means iterations instead of just 1 before balancing
- This should produce tighter, more natural clusters that require less
  aggressive balancing adjustments

Hypothesis: The parent's single k-means iteration may not converge centroids
to optimal positions. Multiple iterations will produce more stable clusters
with better initial balance, requiring fewer disruptive swaps during
balancing. This should improve both compactness and balance scores.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Greedy centroid clustering with multiple k-means iterations."""
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

    # MUTATION: Multiple k-means iterations (5 instead of 1)
    assignments = [-1] * n
    rep_accounts = [[] for _ in range(num_reps)]
    rep_revenues = [0] * num_reps

    for kmeans_iter in range(5):
        # Reset assignments
        for rep_id in range(num_reps):
            rep_accounts[rep_id] = []
            rep_revenues[rep_id] = 0

        # Assign each account to nearest centroid
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

    # Revenue balancing: greedy swaps between most/least balanced territories
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    for iteration in range(100):
        current_cv = compute_cv()
        if current_cv < 0.05:  # Good enough balance (5% CV)
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

        if best_action is None or best_improvement < 0.001:
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
