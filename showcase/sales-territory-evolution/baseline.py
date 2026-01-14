"""
Baseline territory assignment using simple k-means style clustering.

This is intentionally simple to leave room for evolution to improve.
It assigns accounts to the nearest rep centroid without considering
revenue balance or optimization.
"""

import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """
    Assign accounts to sales reps using simple k-means clustering.

    Strategy:
    1. Initialize rep centroids by spreading them across the account locations
    2. Assign each account to nearest centroid
    3. Update centroids based on assignments
    4. Repeat for a few iterations

    Args:
        accounts: List of dicts with keys: id, lat, lon, revenue, industry
        num_reps: Number of sales reps

    Returns:
        Dict mapping rep_id (0 to num_reps-1) -> list of account_ids
    """
    if not accounts or num_reps <= 0:
        return {i: [] for i in range(num_reps)}

    # Initialize centroids by picking spread-out accounts
    sorted_by_lat = sorted(accounts, key=lambda a: a["lat"])
    step = len(sorted_by_lat) // num_reps
    centroids = []
    for i in range(num_reps):
        idx = min(i * step + step // 2, len(sorted_by_lat) - 1)
        centroids.append((sorted_by_lat[idx]["lat"], sorted_by_lat[idx]["lon"]))

    # Run k-means for 10 iterations
    assignments = {i: [] for i in range(num_reps)}

    for _ in range(10):
        # Clear assignments
        assignments = {i: [] for i in range(num_reps)}

        # Assign each account to nearest centroid
        for account in accounts:
            min_dist = float("inf")
            best_rep = 0
            for rep_id, centroid in enumerate(centroids):
                dist = haversine_distance(account["lat"], account["lon"], centroid[0], centroid[1])
                if dist < min_dist:
                    min_dist = dist
                    best_rep = rep_id
            assignments[best_rep].append(account["id"])

        # Update centroids
        new_centroids = []
        for rep_id in range(num_reps):
            rep_accounts = [a for a in accounts if a["id"] in assignments[rep_id]]
            if rep_accounts:
                avg_lat = sum(a["lat"] for a in rep_accounts) / len(rep_accounts)
                avg_lon = sum(a["lon"] for a in rep_accounts) / len(rep_accounts)
                new_centroids.append((avg_lat, avg_lon))
            else:
                new_centroids.append(centroids[rep_id])
        centroids = new_centroids

    return assignments


if __name__ == "__main__":
    from data import get_accounts
    import json

    accounts = get_accounts()
    result = assign_territories(accounts, num_reps=4)

    print("Territory Assignments:")
    for rep_id, account_ids in result.items():
        rep_accounts = [a for a in accounts if a["id"] in account_ids]
        total_rev = sum(a["revenue"] for a in rep_accounts)
        print(f"  Rep {rep_id}: {len(account_ids)} accounts, ${total_rev:,} revenue")
        for a in rep_accounts:
            print(f"    - {a['name']} ({a['industry']}): ${a['revenue']:,}")
