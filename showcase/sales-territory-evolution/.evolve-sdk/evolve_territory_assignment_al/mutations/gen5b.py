"""
Variant gen5b: Multi-pair balance optimization with compactness bonus.

Mutation from gen3b: Add compactness consideration to move/swap evaluation.
Instead of purely optimizing CV (balance), include a bonus for moves that
improve geographic compactness by keeping accounts closer to their territory
centroid.

Key changes:
1. Track centroids for each territory during balancing phase
2. Add compactness term to move/swap scoring: favor moves where the account
   ends up closer to the target territory's centroid
3. Combined scoring: balance_improvement + compactness_factor * compactness_bonus

Hypothesis: gen3b achieves excellent balance (0.9914) but poor compactness (0.7422).
By adding a compactness bonus, we can make geographically smarter move choices
that improve compactness while maintaining good balance. The compactness factor
is small (0.0001) so balance remains primary but tie-breaks favor compact moves.
"""

import math


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Greedy centroid clustering with compactness-aware multi-pair balance optimization."""
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

    # MUTATION: Multi-pair balance optimization WITH COMPACTNESS BONUS
    # Evaluate moves/swaps on combined balance + compactness score

    def compute_cv():
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
        return math.sqrt(variance) / mean_rev

    def update_centroids():
        """Update centroids based on current assignments."""
        for rep_id in range(num_reps):
            if rep_accounts[rep_id]:
                centroids_lat[rep_id] = sum(lats[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])
                centroids_lon[rep_id] = sum(lons[i] for i in rep_accounts[rep_id]) / len(rep_accounts[rep_id])

    def dist_to_centroid(account_idx, rep_id):
        """Squared distance from account to territory centroid."""
        return (lats[account_idx] - centroids_lat[rep_id]) ** 2 + (lons[account_idx] - centroids_lon[rep_id]) ** 2

    # Compactness factor: weight for compactness vs balance trade-off
    # Small value ensures balance is primary, but tie-breaks favor compact moves
    compactness_factor = 0.0001

    for iteration in range(150):  # More iterations for multi-pair exploration
        current_cv = compute_cv()
        if current_cv < 0.02:  # Tighter target: 2% CV
            break

        # Update centroids for compactness calculation
        update_centroids()

        # Find all over-target and under-target territories
        over_target = [r for r in range(num_reps) if rep_revenues[r] > target_revenue]
        under_target = [r for r in range(num_reps) if rep_revenues[r] < target_revenue]

        if not over_target or not under_target:
            break

        best_score = 0
        best_action = None

        # Try moves/swaps between ALL over->under pairs
        for from_rep in over_target:
            for to_rep in under_target:
                # Try moving accounts from from_rep to to_rep
                for i in rep_accounts[from_rep]:
                    if len(rep_accounts[from_rep]) <= 1:
                        continue  # Don't empty a territory

                    new_revs = rep_revenues[:]
                    new_revs[from_rep] -= revenues[i]
                    new_revs[to_rep] += revenues[i]

                    new_mean = sum(new_revs) / num_reps
                    new_var = sum((r - new_mean) ** 2 for r in new_revs) / num_reps
                    new_cv = math.sqrt(new_var) / new_mean if new_mean > 0 else 0

                    balance_improvement = current_cv - new_cv

                    # Compactness bonus: reward moves where account is closer to target centroid
                    old_dist = dist_to_centroid(i, from_rep)
                    new_dist = dist_to_centroid(i, to_rep)
                    # Positive bonus if moving closer, negative if moving farther
                    compactness_bonus = old_dist - new_dist

                    combined_score = balance_improvement + compactness_factor * compactness_bonus

                    if combined_score > best_score:
                        best_score = combined_score
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

                        balance_improvement = current_cv - new_cv

                        # Compactness bonus for swap: sum of distance improvements for both accounts
                        old_dist_i = dist_to_centroid(i, from_rep)
                        new_dist_i = dist_to_centroid(i, to_rep)
                        old_dist_j = dist_to_centroid(j, to_rep)
                        new_dist_j = dist_to_centroid(j, from_rep)
                        # Net compactness improvement from swapping
                        compactness_bonus = (old_dist_i - new_dist_i) + (old_dist_j - new_dist_j)

                        combined_score = balance_improvement + compactness_factor * compactness_bonus

                        if combined_score > best_score:
                            best_score = combined_score
                            best_action = ("swap", i, j, from_rep, to_rep)

        if best_action is None or best_score < 0.0005:
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
