"""
Variant gen1a: Compactness-first clustering with balanced seeding.

Mutation from gen0_a: Algorithm family change - Replace k-means iterations
with a single-pass nearest-centroid assignment that uses geographic seeds
distributed around account clusters.

Key change: Use actual account cluster centers as seeds (not centroid of mass)
and perform single-pass assignment with immediate centroid updates for better
geographic compactness.

Hypothesis: The k-means iterations in parent drift centroids away from
natural clusters. Single-pass with immediate updates keeps assignments tight.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Compactness-first with balanced post-processing."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-compute account data for faster access
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    # Initialize seeds using k-means++ style selection for maximal spread
    seeds_lat = []
    seeds_lon = []
    used = set()

    # First seed: account with most neighbors (cluster center proxy)
    neighbor_counts = []
    for i in range(n):
        count = sum(1 for j in range(n) if i != j and
                   (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2 < 0.01)
        neighbor_counts.append(count)

    first_idx = max(range(n), key=lambda i: neighbor_counts[i])
    seeds_lat.append(lats[first_idx])
    seeds_lon.append(lons[first_idx])
    used.add(first_idx)

    # Remaining seeds: max-min distance selection
    for _ in range(1, num_reps):
        max_min_dist = -1
        best_idx = 0
        for i in range(n):
            if i in used:
                continue
            min_dist = min(
                (lats[i] - slat)**2 + (lons[i] - slon)**2
                for slat, slon in zip(seeds_lat, seeds_lon)
            )
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        used.add(best_idx)
        seeds_lat.append(lats[best_idx])
        seeds_lon.append(lons[best_idx])

    # Single-pass assignment: assign closest unassigned to nearest seed
    # with immediate centroid update for tighter clusters
    assignments = [-1] * n
    territory_lats = [[] for _ in range(num_reps)]
    territory_lons = [[] for _ in range(num_reps)]
    territory_revs = [0] * num_reps

    # Sort accounts by distance to nearest seed (closest first)
    def min_seed_dist(i):
        return min((lats[i] - slat)**2 + (lons[i] - slon)**2
                   for slat, slon in zip(seeds_lat, seeds_lon))

    sorted_indices = sorted(range(n), key=min_seed_dist)

    centroid_lats = seeds_lat[:]
    centroid_lons = seeds_lon[:]

    for i in sorted_indices:
        # Find nearest centroid
        min_dist = float("inf")
        best_rep = 0
        for rep_id in range(num_reps):
            dist = (lats[i] - centroid_lats[rep_id])**2 + (lons[i] - centroid_lons[rep_id])**2
            if dist < min_dist:
                min_dist = dist
                best_rep = rep_id

        assignments[i] = best_rep
        territory_lats[best_rep].append(lats[i])
        territory_lons[best_rep].append(lons[i])
        territory_revs[best_rep] += revenues[i]

        # Update centroid immediately (incremental mean)
        k = len(territory_lats[best_rep])
        centroid_lats[best_rep] = sum(territory_lats[best_rep]) / k
        centroid_lons[best_rep] = sum(territory_lons[best_rep]) / k

    # Revenue balancing via targeted swaps
    def revenue_cv(rev_list):
        mean_rev = sum(rev_list) / len(rev_list)
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rev_list) / len(rev_list)
        return math.sqrt(variance) / mean_rev

    # Greedy swap optimization - only between most imbalanced pairs
    for _ in range(50):
        rep_revenues = territory_revs[:]
        current_cv = revenue_cv(rep_revenues)

        if current_cv < 0.01:  # Already well balanced
            break

        best_improvement = 0
        best_swap = None

        # Find most and least revenue territories
        max_rep = max(range(num_reps), key=lambda r: rep_revenues[r])
        min_rep = min(range(num_reps), key=lambda r: rep_revenues[r])

        if max_rep == min_rep:
            break

        max_rep_accounts = [i for i in range(n) if assignments[i] == max_rep]
        min_rep_accounts = [i for i in range(n) if assignments[i] == min_rep]

        # Try moves from max to min
        for i in max_rep_accounts:
            new_rev = rep_revenues[:]
            new_rev[max_rep] -= revenues[i]
            new_rev[min_rep] += revenues[i]
            new_cv = revenue_cv(new_rev)
            improvement = current_cv - new_cv
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = ("move", i, min_rep)

        # Try swaps between max and min
        for i in max_rep_accounts:
            for j in min_rep_accounts:
                new_rev = rep_revenues[:]
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
            old_rep = assignments[i]
            assignments[i] = new_rep
            territory_revs[old_rep] -= revenues[i]
            territory_revs[new_rep] += revenues[i]
        else:
            _, i, j = best_swap
            rep_i, rep_j = assignments[i], assignments[j]
            assignments[i], assignments[j] = rep_j, rep_i
            territory_revs[rep_i] = territory_revs[rep_i] - revenues[i] + revenues[j]
            territory_revs[rep_j] = territory_revs[rep_j] - revenues[j] + revenues[i]

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
