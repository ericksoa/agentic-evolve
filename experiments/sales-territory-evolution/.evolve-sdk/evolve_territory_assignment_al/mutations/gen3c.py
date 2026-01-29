"""
Variant gen3c: Eliminate O(n^2) neighbor counting, add compactness-aware balancing.

Mutation from gen1a: Algorithm optimization - Remove the O(n^2) neighbor counting
loop used for first seed selection (marginal benefit, high cost). Replace with
simpler extremal point selection (min/max lat). Add compactness-weighted term to
the balance swap decisions.

Key changes:
1. Replace O(n^2) neighbor counting with O(n) extremal point selection
2. Add distance penalty to balance swaps - prefer moving accounts closer to their
   new territory's centroid
3. Pre-compute and reuse centroid positions during swap phase

Hypothesis: The O(n^2) neighbor counting contributes minimal clustering quality
but is expensive. Extremal points provide good spread with O(n) cost. Adding
distance awareness to swaps will improve compactness (0.7651 -> target 0.82+)
without sacrificing the excellent balance score (0.9956).
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Compactness-aware clustering with efficient seeding."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-compute account data for faster access
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    # Precompute squared distances for all pairs (avoid repeated computation)
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    # MUTATION: Replace O(n^2) neighbor counting with O(n) extremal selection
    # First seed: account with minimum latitude (deterministic, good spread start)
    seeds_lat = []
    seeds_lon = []
    used = set()

    first_idx = min(range(n), key=lambda i: lats[i])
    seeds_lat.append(lats[first_idx])
    seeds_lon.append(lons[first_idx])
    used.add(first_idx)

    # Remaining seeds: max-min distance selection (k-means++ style)
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

    # Single-pass assignment with immediate centroid update
    assignments = [-1] * n
    territory_lats = [[] for _ in range(num_reps)]
    territory_lons = [[] for _ in range(num_reps)]
    territory_revs = [0] * num_reps

    # Sort by distance to nearest seed (closest first for tighter initial clusters)
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

    # Build territory account lists for swap phase
    territory_accounts = [[] for _ in range(num_reps)]
    for i in range(n):
        territory_accounts[assignments[i]].append(i)

    # Recompute centroids from final assignments
    for rep_id in range(num_reps):
        if territory_accounts[rep_id]:
            centroid_lats[rep_id] = sum(lats[i] for i in territory_accounts[rep_id]) / len(territory_accounts[rep_id])
            centroid_lons[rep_id] = sum(lons[i] for i in territory_accounts[rep_id]) / len(territory_accounts[rep_id])

    # MUTATION: Compactness-aware revenue balancing
    # Add distance penalty to swap decisions to improve geographic tightness
    def revenue_cv(rev_list):
        mean_rev = sum(rev_list) / len(rev_list)
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rev_list) / len(rev_list)
        return math.sqrt(variance) / mean_rev

    def dist_to_centroid(acct_idx, rep_id):
        return (lats[acct_idx] - centroid_lats[rep_id])**2 + (lons[acct_idx] - centroid_lons[rep_id])**2

    # Greedy swap with compactness awareness
    for _ in range(60):  # Increased iterations for better exploration
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

        # Try moves from max to min with compactness penalty
        for i in territory_accounts[max_rep]:
            if len(territory_accounts[max_rep]) <= 1:
                continue  # Don't empty a territory

            new_rev = rep_revenues[:]
            new_rev[max_rep] -= revenues[i]
            new_rev[min_rep] += revenues[i]
            new_cv = revenue_cv(new_rev)
            balance_improvement = current_cv - new_cv

            # MUTATION: Compactness term - prefer moves that reduce distance to new centroid
            dist_gain = dist_to_centroid(i, max_rep) - dist_to_centroid(i, min_rep)
            # Normalize: positive dist_gain means moving closer, which is good
            compactness_bonus = dist_gain * 0.0001 if dist_gain > 0 else dist_gain * 0.0002

            combined_improvement = balance_improvement + compactness_bonus

            if combined_improvement > best_improvement:
                best_improvement = combined_improvement
                best_swap = ("move", i, min_rep)

        # Try swaps between max and min with compactness awareness
        for i in territory_accounts[max_rep]:
            for j in territory_accounts[min_rep]:
                new_rev = rep_revenues[:]
                new_rev[max_rep] = new_rev[max_rep] - revenues[i] + revenues[j]
                new_rev[min_rep] = new_rev[min_rep] - revenues[j] + revenues[i]
                new_cv = revenue_cv(new_rev)
                balance_improvement = current_cv - new_cv

                # Compactness: each account should be closer to its new territory
                dist_gain_i = dist_to_centroid(i, max_rep) - dist_to_centroid(i, min_rep)
                dist_gain_j = dist_to_centroid(j, min_rep) - dist_to_centroid(j, max_rep)
                compactness_bonus = (dist_gain_i + dist_gain_j) * 0.00005

                combined_improvement = balance_improvement + compactness_bonus

                if combined_improvement > best_improvement:
                    best_improvement = combined_improvement
                    best_swap = ("swap", i, j)

        if best_swap is None or best_improvement < 0.0005:
            break

        if best_swap[0] == "move":
            _, i, new_rep = best_swap
            old_rep = assignments[i]
            assignments[i] = new_rep
            territory_accounts[old_rep].remove(i)
            territory_accounts[new_rep].append(i)
            territory_revs[old_rep] -= revenues[i]
            territory_revs[new_rep] += revenues[i]
            # Update centroids
            if territory_accounts[old_rep]:
                centroid_lats[old_rep] = sum(lats[k] for k in territory_accounts[old_rep]) / len(territory_accounts[old_rep])
                centroid_lons[old_rep] = sum(lons[k] for k in territory_accounts[old_rep]) / len(territory_accounts[old_rep])
            centroid_lats[new_rep] = sum(lats[k] for k in territory_accounts[new_rep]) / len(territory_accounts[new_rep])
            centroid_lons[new_rep] = sum(lons[k] for k in territory_accounts[new_rep]) / len(territory_accounts[new_rep])
        else:
            _, i, j = best_swap
            rep_i, rep_j = assignments[i], assignments[j]
            assignments[i], assignments[j] = rep_j, rep_i
            territory_accounts[rep_i].remove(i)
            territory_accounts[rep_i].append(j)
            territory_accounts[rep_j].remove(j)
            territory_accounts[rep_j].append(i)
            territory_revs[rep_i] = territory_revs[rep_i] - revenues[i] + revenues[j]
            territory_revs[rep_j] = territory_revs[rep_j] - revenues[j] + revenues[i]
            # Update centroids
            centroid_lats[rep_i] = sum(lats[k] for k in territory_accounts[rep_i]) / len(territory_accounts[rep_i])
            centroid_lons[rep_i] = sum(lons[k] for k in territory_accounts[rep_i]) / len(territory_accounts[rep_i])
            centroid_lats[rep_j] = sum(lats[k] for k in territory_accounts[rep_j]) / len(territory_accounts[rep_j])
            centroid_lons[rep_j] = sum(lons[k] for k in territory_accounts[rep_j]) / len(territory_accounts[rep_j])

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
