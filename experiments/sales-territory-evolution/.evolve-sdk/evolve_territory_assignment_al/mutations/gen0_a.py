"""
Variant A: K-means clustering with revenue-balancing swaps.

Approach:
1. Use k-means to create geographically compact initial clusters
2. Apply greedy swap optimization to balance revenue across territories
3. Uses Euclidean distance approximation for faster computation

Performance optimizations:
- Pre-compute distance matrix once
- Use squared distances to avoid sqrt in inner loop
- Early termination when no improvement found
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """K-means with revenue-balancing post-processing."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-compute account data for faster access
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    # Pre-compute squared distance matrix (use Euclidean for speed)
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dlat = lats[i] - lats[j]
            dlon = lons[i] - lons[j]
            d = dlat * dlat + dlon * dlon
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    # Initialize centroids using k-means++ style selection
    centroid_lats = []
    centroid_lons = []

    # First centroid: account closest to center of mass
    avg_lat = sum(lats) / n
    avg_lon = sum(lons) / n
    first_idx = min(range(n), key=lambda i: (lats[i] - avg_lat)**2 + (lons[i] - avg_lon)**2)
    centroid_lats.append(lats[first_idx])
    centroid_lons.append(lons[first_idx])

    # Remaining centroids: pick accounts far from existing centroids
    used = {first_idx}
    for _ in range(1, num_reps):
        max_min_dist = -1
        best_idx = 0
        for i in range(n):
            if i in used:
                continue
            min_dist_to_centroid = min(
                (lats[i] - clat)**2 + (lons[i] - clon)**2
                for clat, clon in zip(centroid_lats, centroid_lons)
            )
            if min_dist_to_centroid > max_min_dist:
                max_min_dist = min_dist_to_centroid
                best_idx = i
        used.add(best_idx)
        centroid_lats.append(lats[best_idx])
        centroid_lons.append(lons[best_idx])

    # K-means iterations
    assignments = [0] * n  # assignment[i] = rep_id for account i

    for iteration in range(15):
        # Assign each account to nearest centroid
        for i in range(n):
            min_dist = float("inf")
            best_rep = 0
            for rep_id in range(num_reps):
                dist = (lats[i] - centroid_lats[rep_id])**2 + (lons[i] - centroid_lons[rep_id])**2
                if dist < min_dist:
                    min_dist = dist
                    best_rep = rep_id
            assignments[i] = best_rep

        # Update centroids
        for rep_id in range(num_reps):
            rep_indices = [i for i in range(n) if assignments[i] == rep_id]
            if rep_indices:
                centroid_lats[rep_id] = sum(lats[i] for i in rep_indices) / len(rep_indices)
                centroid_lons[rep_id] = sum(lons[i] for i in rep_indices) / len(rep_indices)

    # Revenue balancing via greedy swaps
    def get_rep_revenues():
        rev = [0] * num_reps
        for i in range(n):
            rev[assignments[i]] += revenues[i]
        return rev

    def revenue_cv(rev_list):
        mean_rev = sum(rev_list) / len(rev_list)
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rev_list) / len(rev_list)
        return math.sqrt(variance) / mean_rev

    # Greedy swap optimization
    for _ in range(50):
        rep_revenues = get_rep_revenues()
        current_cv = revenue_cv(rep_revenues)

        best_improvement = 0
        best_swap = None

        # Find best swap between highest and lowest revenue reps
        max_rep = max(range(num_reps), key=lambda r: rep_revenues[r])
        min_rep = min(range(num_reps), key=lambda r: rep_revenues[r])

        if max_rep == min_rep:
            break

        max_rep_accounts = [i for i in range(n) if assignments[i] == max_rep]
        min_rep_accounts = [i for i in range(n) if assignments[i] == min_rep]

        # Try single moves from max_rep to min_rep
        for i in max_rep_accounts:
            new_rev = rep_revenues.copy()
            new_rev[max_rep] -= revenues[i]
            new_rev[min_rep] += revenues[i]
            new_cv = revenue_cv(new_rev)
            improvement = current_cv - new_cv
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = ("move", i, min_rep)

        # Try swaps between max_rep and min_rep
        for i in max_rep_accounts:
            for j in min_rep_accounts:
                new_rev = rep_revenues.copy()
                new_rev[max_rep] = new_rev[max_rep] - revenues[i] + revenues[j]
                new_rev[min_rep] = new_rev[min_rep] - revenues[j] + revenues[i]
                new_cv = revenue_cv(new_rev)
                improvement = current_cv - new_cv
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = ("swap", i, j)

        if best_swap is None or best_improvement < 0.001:
            break

        if best_swap[0] == "move":
            _, i, new_rep = best_swap
            assignments[i] = new_rep
        else:
            _, i, j = best_swap
            assignments[i], assignments[j] = assignments[j], assignments[i]

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
