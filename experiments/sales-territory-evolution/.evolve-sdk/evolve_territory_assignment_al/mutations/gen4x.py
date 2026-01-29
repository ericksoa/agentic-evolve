"""
Variant gen4x: Crossover combining best features from gen3x, gen3b, gen1b.

Parents:
- gen3x (0.8168): Precomputed distances, K-means++ seeding, sorted assignment with
  incremental centroid updates, revenue-aware scoring with overfill penalties
- gen3b (0.8097): Multi-pair balance optimization (evaluates ALL over/under pairs),
  finer improvement thresholds, more balancing iterations
- gen1b (0.7993): Precomputed sorted neighbor lists for efficient geographic
  candidate selection during balancing

Crossover design:
1. From gen3x: Precomputed squared distance matrix for O(1) lookups
2. From gen3x: K-means++ seed initialization for maximum spread
3. From gen3x: Sorted assignment order (closest first) with incremental centroid updates
4. From gen3x: Revenue-aware scoring with overfill penalties during assignment
5. From gen3b: Multi-pair balance optimization - evaluates ALL over/under territory pairs
6. From gen3b: Finer improvement threshold (0.0005) and more iterations (150)
7. From gen1b: Precomputed sorted neighbor lists to prioritize geographically close
   move/swap candidates during balancing (improves compactness)

Hypothesis: Combining the best assignment phase from gen3x with gen3b's exhaustive
multi-pair exploration and gen1b's neighbor-aware candidate selection should yield
both better revenue balance and improved compactness.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid clustering with precomputed distances and multi-pair neighbor-aware balancing."""
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

    # From gen3x + gen1b: Precompute all pairwise squared distances for O(1) lookups
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = (lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2
            dist_sq[i][j] = d_sq
            dist_sq[j][i] = d_sq

    # From gen1b: Precompute sorted neighbor lists for each account (nearest to farthest)
    # Used later to prioritize geographically close swap candidates
    neighbors = []
    for i in range(n):
        neighbor_list = [(j, dist_sq[i][j]) for j in range(n) if j != i]
        neighbor_list.sort(key=lambda x: x[1])
        neighbors.append([x[0] for x in neighbor_list])

    # From gen3x: K-means++ style seed initialization for maximum spread
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
        # Find best territory considering distance and revenue balance
        best_score = float("inf")
        best_rep = 0

        for rep_id in range(num_reps):
            # Distance to current centroid
            dist = (lats[i] - centroid_lats[rep_id]) ** 2 + (lons[i] - centroid_lons[rep_id]) ** 2

            # Revenue balance factor (from gen3x: score with revenue bias)
            rev_deviation = abs(rep_revenues[rep_id] + revenues[i] - target_revenue)
            rev_factor = rev_deviation * 0.00001  # Small weight to break ties

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

    # From gen3b + gen1b: Multi-pair balance optimization with neighbor-aware candidates
    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    # From gen3b: More iterations for thorough multi-pair exploration
    for iteration in range(150):
        current_cv = compute_cv()
        if current_cv < 0.02:  # 2% CV target
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
                # From gen1b: Sort candidates by proximity to target territory
                # This prioritizes moves that maintain compactness
                if rep_accounts[to_rep]:
                    # Sort from_rep accounts by min distance to any account in to_rep
                    candidates = sorted(
                        rep_accounts[from_rep],
                        key=lambda idx: min(dist_sq[idx][other] for other in rep_accounts[to_rep])
                    )
                else:
                    candidates = rep_accounts[from_rep]

                # Try moving accounts (prioritize geographically close ones)
                for i in candidates:
                    if len(rep_accounts[from_rep]) <= 1:
                        continue  # Don't empty a territory

                    new_revs = rep_revenues[:]
                    new_revs[from_rep] -= revenues[i]
                    new_revs[to_rep] += revenues[i]

                    new_mean = sum(new_revs) / num_reps
                    new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                    new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                    improvement = current_cv - new_cv

                    # From gen1b/gen3x: Compactness bonus for moving to nearby accounts
                    if rep_accounts[to_rep]:
                        min_dist_to_target = min(dist_sq[i][other] for other in rep_accounts[to_rep])
                        compactness_bonus = 0.001 / (1.0 + min_dist_to_target)
                        improvement += compactness_bonus

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_action = ("move", i, from_rep, to_rep)

                # Try swapping accounts between from_rep and to_rep
                for i in candidates[:len(candidates)//2 + 1]:  # Limit swap candidates for efficiency
                    for j in rep_accounts[to_rep]:
                        new_revs = rep_revenues[:]
                        new_revs[from_rep] = new_revs[from_rep] - revenues[i] + revenues[j]
                        new_revs[to_rep] = new_revs[to_rep] - revenues[j] + revenues[i]

                        new_mean = sum(new_revs) / num_reps
                        new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                        new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                        improvement = current_cv - new_cv

                        # Compactness consideration for swaps
                        if improvement > 0 and rep_accounts[to_rep] and rep_accounts[from_rep]:
                            # Bonus if i is closer to to_rep's accounts and j is closer to from_rep's
                            dist_i_to_to = min(dist_sq[i][other] for other in rep_accounts[to_rep] if other != j)  if len(rep_accounts[to_rep]) > 1 else 0
                            dist_j_to_from = min(dist_sq[j][other] for other in rep_accounts[from_rep] if other != i) if len(rep_accounts[from_rep]) > 1 else 0
                            swap_compactness = 0.0005 / (1.0 + dist_i_to_to + dist_j_to_from)
                            improvement += swap_compactness

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_action = ("swap", i, j, from_rep, to_rep)

        # From gen3b: Finer threshold for accepting improvements
        if best_action is None or best_improvement < 0.0005:
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
