"""
Variant gen2c: Compactness-aware revenue balancing.

Mutation from gen1a: Add distance penalty to swap optimization phase.
The parent has excellent balance (0.9956) but weak compactness (0.7651).
This mutation adds a compactness penalty to the revenue balancing swaps,
so moves/swaps that hurt geographic compactness are penalized.

Key change: During the greedy swap optimization, penalize moves that
increase the average distance from accounts to their territory centroids.

Hypothesis: The parent's swap phase moves accounts purely for revenue
balance without considering geography, causing compactness degradation.
Adding a distance penalty will preserve balance while improving compactness.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Compactness-first with compactness-aware balancing."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-compute account data for faster access
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    # Precompute squared distances to avoid sqrt overhead
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    # Initialize seeds using k-means++ style selection for maximal spread
    seeds_lat = []
    seeds_lon = []
    used = set()

    # First seed: account with most neighbors (cluster center proxy)
    neighbor_counts = []
    for i in range(n):
        count = sum(1 for j in range(n) if i != j and dist_sq[i][j] < 0.01)
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
    territory_indices = [[] for _ in range(num_reps)]
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
        territory_indices[best_rep].append(i)
        territory_revs[best_rep] += revenues[i]

        # Update centroid immediately (incremental mean)
        k = len(territory_indices[best_rep])
        centroid_lats[best_rep] = sum(lats[idx] for idx in territory_indices[best_rep]) / k
        centroid_lons[best_rep] = sum(lons[idx] for idx in territory_indices[best_rep]) / k

    # Helper: compute average squared distance to centroid for a territory
    def territory_compactness_cost(rep_id):
        indices = territory_indices[rep_id]
        if not indices:
            return 0
        cx = sum(lats[i] for i in indices) / len(indices)
        cy = sum(lons[i] for i in indices) / len(indices)
        return sum((lats[i] - cx)**2 + (lons[i] - cy)**2 for i in indices)

    # Revenue balancing via targeted swaps with compactness penalty
    def revenue_cv(rev_list):
        mean_rev = sum(rev_list) / len(rev_list)
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rev_list) / len(rev_list)
        return math.sqrt(variance) / mean_rev

    # Compactness weight: controls balance vs compactness tradeoff
    # Higher weight preserves compactness better at cost of some balance
    COMPACTNESS_WEIGHT = 0.05

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

        max_rep_accounts = [i for i in territory_indices[max_rep]]
        min_rep_accounts = [i for i in territory_indices[min_rep]]

        # Precompute current compactness costs for affected territories
        max_rep_compact = territory_compactness_cost(max_rep)
        min_rep_compact = territory_compactness_cost(min_rep)
        current_compact_cost = max_rep_compact + min_rep_compact

        # Try moves from max to min
        for i in max_rep_accounts:
            new_rev = rep_revenues[:]
            new_rev[max_rep] -= revenues[i]
            new_rev[min_rep] += revenues[i]
            new_cv = revenue_cv(new_rev)
            cv_improvement = current_cv - new_cv

            # Estimate compactness change for this move
            # Account i moves from max_rep to min_rep
            # Simple proxy: distance to min_rep centroid vs max_rep centroid
            dist_to_max = (lats[i] - centroid_lats[max_rep])**2 + (lons[i] - centroid_lons[max_rep])**2
            dist_to_min = (lats[i] - centroid_lats[min_rep])**2 + (lons[i] - centroid_lons[min_rep])**2
            compact_penalty = (dist_to_min - dist_to_max) * COMPACTNESS_WEIGHT

            combined_improvement = cv_improvement - compact_penalty

            if combined_improvement > best_improvement:
                best_improvement = combined_improvement
                best_swap = ("move", i, min_rep)

        # Try swaps between max and min
        for i in max_rep_accounts:
            for j in min_rep_accounts:
                new_rev = rep_revenues[:]
                new_rev[max_rep] = new_rev[max_rep] - revenues[i] + revenues[j]
                new_rev[min_rep] = new_rev[min_rep] - revenues[j] + revenues[i]
                new_cv = revenue_cv(new_rev)
                cv_improvement = current_cv - new_cv

                # Compactness penalty for swap: i goes to min_rep, j goes to max_rep
                dist_i_to_max = (lats[i] - centroid_lats[max_rep])**2 + (lons[i] - centroid_lons[max_rep])**2
                dist_i_to_min = (lats[i] - centroid_lats[min_rep])**2 + (lons[i] - centroid_lons[min_rep])**2
                dist_j_to_max = (lats[j] - centroid_lats[max_rep])**2 + (lons[j] - centroid_lons[max_rep])**2
                dist_j_to_min = (lats[j] - centroid_lats[min_rep])**2 + (lons[j] - centroid_lons[min_rep])**2

                # Net compactness change
                compact_penalty = ((dist_i_to_min - dist_i_to_max) + (dist_j_to_max - dist_j_to_min)) * COMPACTNESS_WEIGHT

                combined_improvement = cv_improvement - compact_penalty

                if combined_improvement > best_improvement:
                    best_improvement = combined_improvement
                    best_swap = ("swap", i, j)

        if best_swap is None or best_improvement < 0.0001:
            break

        if best_swap[0] == "move":
            _, i, new_rep = best_swap
            old_rep = assignments[i]
            assignments[i] = new_rep
            territory_indices[old_rep].remove(i)
            territory_indices[new_rep].append(i)
            territory_revs[old_rep] -= revenues[i]
            territory_revs[new_rep] += revenues[i]
            # Update centroids
            if territory_indices[old_rep]:
                centroid_lats[old_rep] = sum(lats[idx] for idx in territory_indices[old_rep]) / len(territory_indices[old_rep])
                centroid_lons[old_rep] = sum(lons[idx] for idx in territory_indices[old_rep]) / len(territory_indices[old_rep])
            if territory_indices[new_rep]:
                centroid_lats[new_rep] = sum(lats[idx] for idx in territory_indices[new_rep]) / len(territory_indices[new_rep])
                centroid_lons[new_rep] = sum(lons[idx] for idx in territory_indices[new_rep]) / len(territory_indices[new_rep])
        else:
            _, i, j = best_swap
            rep_i, rep_j = assignments[i], assignments[j]
            assignments[i], assignments[j] = rep_j, rep_i
            territory_indices[rep_i].remove(i)
            territory_indices[rep_i].append(j)
            territory_indices[rep_j].remove(j)
            territory_indices[rep_j].append(i)
            territory_revs[rep_i] = territory_revs[rep_i] - revenues[i] + revenues[j]
            territory_revs[rep_j] = territory_revs[rep_j] - revenues[j] + revenues[i]
            # Update centroids
            centroid_lats[rep_i] = sum(lats[idx] for idx in territory_indices[rep_i]) / len(territory_indices[rep_i])
            centroid_lons[rep_i] = sum(lons[idx] for idx in territory_indices[rep_i]) / len(territory_indices[rep_i])
            centroid_lats[rep_j] = sum(lats[idx] for idx in territory_indices[rep_j]) / len(territory_indices[rep_j])
            centroid_lons[rep_j] = sum(lons[idx] for idx in territory_indices[rep_j]) / len(territory_indices[rep_j])

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
