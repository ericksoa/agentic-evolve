"""
Variant B: Revenue-balanced greedy assignment.

Approach:
1. Sort accounts by revenue (descending - largest first)
2. Always assign to the rep with lowest current revenue (min-heap style)
3. After initial assignment, apply local search to improve compactness

Performance optimizations:
- Single-pass assignment with O(n * num_reps) complexity
- Incremental centroid tracking
- Simple but effective balance-first strategy
"""


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """Revenue-balanced greedy assignment."""
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    n = len(accounts)

    # Pre-extract account data
    account_data = [
        (i, a["id"], a["lat"], a["lon"], a["revenue"])
        for i, a in enumerate(accounts)
    ]

    # Sort by revenue descending (assign large accounts first for better balance)
    sorted_accounts = sorted(account_data, key=lambda x: -x[4])

    # Initialize territory tracking
    territory_ids = [[] for _ in range(num_reps)]
    territory_revenues = [0] * num_reps
    territory_lats = [[] for _ in range(num_reps)]
    territory_lons = [[] for _ in range(num_reps)]

    # Phase 1: Assign accounts to minimize revenue imbalance
    for idx, aid, lat, lon, revenue in sorted_accounts:
        # Find rep with minimum current revenue
        min_rev = float("inf")
        best_rep = 0
        for rep_id in range(num_reps):
            if territory_revenues[rep_id] < min_rev:
                min_rev = territory_revenues[rep_id]
                best_rep = rep_id

        # Assign to rep with lowest revenue
        territory_ids[best_rep].append(aid)
        territory_revenues[best_rep] += revenue
        territory_lats[best_rep].append(lat)
        territory_lons[best_rep].append(lon)

    # Phase 2: Local search to improve compactness while maintaining balance
    # Build index mapping for quick lookups
    assignments = {}  # aid -> rep_id
    account_info = {}  # aid -> (lat, lon, revenue)
    for idx, aid, lat, lon, revenue in account_data:
        account_info[aid] = (lat, lon, revenue)

    for rep_id in range(num_reps):
        for aid in territory_ids[rep_id]:
            assignments[aid] = rep_id

    def get_centroid(rep_id):
        if not territory_lats[rep_id]:
            return 0, 0
        return (
            sum(territory_lats[rep_id]) / len(territory_lats[rep_id]),
            sum(territory_lons[rep_id]) / len(territory_lons[rep_id])
        )

    def dist_to_centroid(lat, lon, rep_id):
        clat, clon = get_centroid(rep_id)
        return (lat - clat)**2 + (lon - clon)**2

    # Try swaps to improve compactness
    for _ in range(30):
        improved = False
        for rep1 in range(num_reps):
            for rep2 in range(rep1 + 1, num_reps):
                for aid1 in list(territory_ids[rep1]):
                    for aid2 in list(territory_ids[rep2]):
                        lat1, lon1, rev1 = account_info[aid1]
                        lat2, lon2, rev2 = account_info[aid2]

                        # Only consider swaps that don't hurt balance too much
                        new_rev1 = territory_revenues[rep1] - rev1 + rev2
                        new_rev2 = territory_revenues[rep2] - rev2 + rev1

                        old_imbalance = abs(territory_revenues[rep1] - territory_revenues[rep2])
                        new_imbalance = abs(new_rev1 - new_rev2)

                        # Check if swap improves compactness
                        old_dist = dist_to_centroid(lat1, lon1, rep1) + dist_to_centroid(lat2, lon2, rep2)
                        new_dist = dist_to_centroid(lat1, lon1, rep2) + dist_to_centroid(lat2, lon2, rep1)

                        # Accept if improves distance without hurting balance too much
                        if new_dist < old_dist * 0.9 and new_imbalance <= old_imbalance * 1.1:
                            # Perform swap
                            territory_ids[rep1].remove(aid1)
                            territory_ids[rep2].remove(aid2)
                            territory_ids[rep1].append(aid2)
                            territory_ids[rep2].append(aid1)

                            # Update tracking
                            territory_revenues[rep1] = new_rev1
                            territory_revenues[rep2] = new_rev2
                            territory_lats[rep1].remove(lat1)
                            territory_lons[rep1].remove(lon1)
                            territory_lats[rep2].remove(lat2)
                            territory_lons[rep2].remove(lon2)
                            territory_lats[rep1].append(lat2)
                            territory_lons[rep1].append(lon2)
                            territory_lats[rep2].append(lat1)
                            territory_lons[rep2].append(lon1)

                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # Convert to output format
    result = {rep_id: territory_ids[rep_id] for rep_id in range(num_reps)}
    return result
