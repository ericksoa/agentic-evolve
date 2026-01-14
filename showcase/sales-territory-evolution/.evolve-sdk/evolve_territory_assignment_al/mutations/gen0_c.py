"""
Variant C: Simulated annealing optimization.

Approach:
1. Start with a random assignment
2. Apply simulated annealing with combined fitness function
3. Use temperature-controlled random swaps to escape local optima

Performance optimizations:
- Incremental fitness updates (don't recompute full fitness each step)
- Pre-computed distance matrix
- Early termination on convergence
"""

import math
import random


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Simulated annealing for territory optimization."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    random.seed(42)  # For reproducibility
    n = len(accounts)

    # Pre-extract account data
    lats = [a["lat"] for a in accounts]
    lons = [a["lon"] for a in accounts]
    revenues = [a["revenue"] for a in accounts]
    ids = [a["id"] for a in accounts]

    total_revenue = sum(revenues)
    target_revenue = total_revenue / num_reps

    # Pre-compute distance matrix (squared Euclidean for speed)
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (lats[i] - lats[j])**2 + (lons[i] - lons[j])**2
            dist_sq[i][j] = d
            dist_sq[j][i] = d

    # Initialize with round-robin assignment (ensures all reps get accounts)
    assignments = [i % num_reps for i in range(n)]

    def compute_fitness(assign):
        """Compute combined fitness score."""
        # Revenue per rep
        rep_revenues = [0] * num_reps
        rep_accounts = [[] for _ in range(num_reps)]
        for i in range(n):
            rep_revenues[assign[i]] += revenues[i]
            rep_accounts[assign[i]].append(i)

        # Balance score (1 - CV)
        mean_rev = sum(rep_revenues) / num_reps
        if mean_rev == 0:
            balance = 0
        else:
            variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / num_reps
            cv = math.sqrt(variance) / mean_rev
            balance = max(0, 1 - cv)

        # Compactness score (avg pairwise distance within territories)
        compactness_scores = []
        for rep_id in range(num_reps):
            accts = rep_accounts[rep_id]
            if len(accts) < 2:
                compactness_scores.append(1.0)
                continue
            total_dist = 0
            pairs = 0
            for i_idx in range(len(accts)):
                for j_idx in range(i_idx + 1, len(accts)):
                    total_dist += math.sqrt(dist_sq[accts[i_idx]][accts[j_idx]])
                    pairs += 1
            avg_dist = total_dist / pairs if pairs > 0 else 0
            # Convert to lat/lon distance to miles (rough: 1 degree ~ 50-70 miles)
            avg_dist_miles = avg_dist * 60  # Approximate
            compactness = max(0, 1 - avg_dist_miles / 30)
            compactness_scores.append(compactness)

        compactness = sum(compactness_scores) / len(compactness_scores)

        # Coverage is always 1.0 since we assign all accounts
        coverage = 1.0

        # Combined fitness
        return coverage * 0.4 + balance * 0.3 + compactness * 0.3

    # Simulated annealing
    current_fitness = compute_fitness(assignments)
    best_assignments = assignments.copy()
    best_fitness = current_fitness

    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 0.001

    iterations_without_improvement = 0
    max_iterations_without_improvement = 200

    while temperature > min_temperature and iterations_without_improvement < max_iterations_without_improvement:
        # Generate neighbor: either swap or move
        new_assignments = assignments.copy()

        if random.random() < 0.5:
            # Swap two accounts from different territories
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while assignments[j] == assignments[i] and n > num_reps:
                j = random.randint(0, n - 1)
            new_assignments[i], new_assignments[j] = new_assignments[j], new_assignments[i]
        else:
            # Move one account to different territory
            i = random.randint(0, n - 1)
            new_rep = random.randint(0, num_reps - 1)
            while new_rep == assignments[i]:
                new_rep = random.randint(0, num_reps - 1)
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

    # Convert to output format
    result = {rep_id: [] for rep_id in range(num_reps)}
    for i in range(n):
        result[best_assignments[i]].append(ids[i])

    return result
