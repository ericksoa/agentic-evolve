"""
Hybrid: K-means initialization with simulated annealing refinement.

Combines approaches from multiple parents:
- K-means++ style centroid initialization (from gen0_a)
- Revenue-aware initial assignment with capacity constraints (from gen0_d)
- Simulated annealing refinement with comprehensive fitness (from gen0_c)

Performance optimizations:
- Pre-computed squared distance matrix
- Efficient incremental neighbor generation
- Combined geographic + revenue scoring in fitness
"""

import math
import random


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Hybrid K-means + simulated annealing territory assignment."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    random.seed(42)  # For reproducibility
    n = len(accounts)

    # Pre-extract account data for faster access
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # Pre-compute squared distance matrix (Euclidean for speed)
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    # =========================================================================
    # PHASE 1: K-means++ style centroid initialization (from gen0_a)
    # =========================================================================
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

    # =========================================================================
    # PHASE 2: Revenue-aware initial assignment (inspired by gen0_d)
    # =========================================================================
    # Assign accounts to nearest centroid with revenue capacity awareness
    assignments = [0] * n
    rep_revenues = [0.0] * num_reps

    # Sort accounts by their distance to nearest centroid (assign closest first)
    account_dist_to_best = []
    for i in range(n):
        distances_to_centroids = [
            (lats[i] - centroid_lats[rep_id])**2 + (lons[i] - centroid_lons[rep_id])**2
            for rep_id in range(num_reps)
        ]
        min_dist = min(distances_to_centroids)
        account_dist_to_best.append((i, min_dist))

    # Assign accounts, prioritizing those closest to centroids
    account_dist_to_best.sort(key=lambda x: x[1])

    for i, _ in account_dist_to_best:
        best_rep = 0
        best_score = float("inf")

        for rep_id in range(num_reps):
            dist = (lats[i] - centroid_lats[rep_id])**2 + (lons[i] - centroid_lons[rep_id])**2

            # Revenue balance penalty (from gen0_d approach)
            rev_penalty = 0
            if rep_revenues[rep_id] > target_revenue * 1.2:
                rev_penalty = (rep_revenues[rep_id] - target_revenue) * 0.0001

            score = dist + rev_penalty

            if score < best_score:
                best_score = score
                best_rep = rep_id

        assignments[i] = best_rep
        rep_revenues[best_rep] += revenues[i]

    # =========================================================================
    # PHASE 3: Simulated annealing refinement (from gen0_c)
    # =========================================================================
    def compute_fitness(assign):
        """Compute combined fitness score (balance + compactness)."""
        # Revenue per rep
        rev_per_rep = [0.0] * num_reps
        rep_accounts = [[] for _ in range(num_reps)]
        for i in range(n):
            rev_per_rep[assign[i]] += revenues[i]
            rep_accounts[assign[i]].append(i)

        # Balance score (1 - CV)
        mean_rev = sum(rev_per_rep) / num_reps
        if mean_rev == 0:
            balance = 0
        else:
            variance = sum((r - mean_rev) ** 2 for r in rev_per_rep) / num_reps
            cv = math.sqrt(variance) / mean_rev
            balance = max(0, 1 - cv)

        # Compactness score: average distance to centroid within territories
        compactness_scores = []
        for rep_id in range(num_reps):
            accts = rep_accounts[rep_id]
            if len(accts) == 0:
                compactness_scores.append(0.0)
                continue
            if len(accts) == 1:
                compactness_scores.append(1.0)
                continue

            # Compute centroid
            c_lat = sum(lats[idx] for idx in accts) / len(accts)
            c_lon = sum(lons[idx] for idx in accts) / len(accts)

            # Average distance to centroid
            total_dist = sum(
                math.sqrt((lats[idx] - c_lat)**2 + (lons[idx] - c_lon)**2)
                for idx in accts
            )
            avg_dist = total_dist / len(accts)
            # Convert to approximate miles (1 degree ~ 60 miles)
            avg_dist_miles = avg_dist * 60
            compactness = max(0, 1 - avg_dist_miles / 30)
            compactness_scores.append(compactness)

        compactness = sum(compactness_scores) / num_reps

        # Coverage is always 1.0 since we assign all accounts
        coverage = 1.0

        # Combined fitness with weights
        return coverage * 0.4 + balance * 0.3 + compactness * 0.3

    # Simulated annealing loop
    current_fitness = compute_fitness(assignments)
    best_assignments = assignments.copy()
    best_fitness = current_fitness

    temperature = 0.5  # Lower starting temp since we have good initial solution
    cooling_rate = 0.995
    min_temperature = 0.001

    iterations_without_improvement = 0
    max_iterations_without_improvement = 150

    while temperature > min_temperature and iterations_without_improvement < max_iterations_without_improvement:
        # Generate neighbor
        new_assignments = assignments.copy()

        if random.random() < 0.5:
            # Swap two accounts from different territories
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            attempts = 0
            while assignments[j] == assignments[i] and attempts < 10:
                j = random.randint(0, n - 1)
                attempts += 1
            if assignments[j] != assignments[i]:
                new_assignments[i], new_assignments[j] = new_assignments[j], new_assignments[i]
        else:
            # Move one account to different territory
            i = random.randint(0, n - 1)
            new_rep = random.randint(0, num_reps - 1)
            if new_rep != assignments[i]:
                new_assignments[i] = new_rep

        new_fitness = compute_fitness(new_assignments)

        # Accept or reject
        delta = new_fitness - current_fitness
        if delta > 0 or random.random() < math.exp(delta / temperature):
            assignments = new_assignments
            current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_assignments = assignments.copy()
                best_fitness = current_fitness
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        else:
            iterations_without_improvement += 1

        temperature *= cooling_rate

    # =========================================================================
    # PHASE 4: Final greedy balancing (from gen0_a)
    # =========================================================================
    assignments = best_assignments

    def get_rep_revenues():
        rev = [0.0] * num_reps
        for i in range(n):
            rev[assignments[i]] += revenues[i]
        return rev

    def revenue_cv(rev_list):
        mean_rev = sum(rev_list) / len(rev_list)
        if mean_rev == 0:
            return 0
        variance = sum((r - mean_rev) ** 2 for r in rev_list) / len(rev_list)
        return math.sqrt(variance) / mean_rev

    for _ in range(30):
        rev_list = get_rep_revenues()
        current_cv = revenue_cv(rev_list)

        if current_cv < 0.05:  # Already well balanced
            break

        max_rep = max(range(num_reps), key=lambda r: rev_list[r])
        min_rep = min(range(num_reps), key=lambda r: rev_list[r])

        if max_rep == min_rep:
            break

        # Find best move from max_rep to min_rep
        best_improvement = 0
        best_move_idx = None

        max_rep_accounts = [i for i in range(n) if assignments[i] == max_rep]

        for i in max_rep_accounts:
            new_rev = rev_list.copy()
            new_rev[max_rep] -= revenues[i]
            new_rev[min_rep] += revenues[i]
            new_cv = revenue_cv(new_rev)
            improvement = current_cv - new_cv
            if improvement > best_improvement:
                best_improvement = improvement
                best_move_idx = i

        if best_move_idx is not None and best_improvement > 0.001:
            assignments[best_move_idx] = min_rep
        else:
            break

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[assignments[i]].append(ids[i])

    return result
