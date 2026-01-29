"""
Synthetic sales account data generator for territory optimization.

Generates accounts clustered around a metro area with realistic attributes.
"""

import random
import math

# Seed for reproducibility
SEED = 42

# Metro area center (fictional city roughly the size of Denver metro)
CENTER_LAT = 39.7392
CENTER_LON = -104.9903

# Industries with typical revenue ranges
INDUSTRIES = [
    ("Technology", 50000, 500000),
    ("Healthcare", 30000, 300000),
    ("Manufacturing", 40000, 400000),
    ("Retail", 10000, 150000),
    ("Financial Services", 60000, 450000),
]


def generate_accounts(n: int = 25, seed: int = SEED) -> list[dict]:
    """
    Generate n synthetic sales accounts.

    Args:
        n: Number of accounts to generate
        seed: Random seed for reproducibility

    Returns:
        List of account dicts with keys: id, lat, lon, revenue, industry, name
    """
    random.seed(seed)
    accounts = []

    # Create 4-5 clusters to simulate business districts
    num_clusters = 4
    cluster_centers = []
    for _ in range(num_clusters):
        # Clusters within ~30 miles of center
        offset_lat = random.gauss(0, 0.15)
        offset_lon = random.gauss(0, 0.15)
        cluster_centers.append((CENTER_LAT + offset_lat, CENTER_LON + offset_lon))

    for i in range(n):
        # Pick a cluster
        cluster = random.choice(cluster_centers)

        # Add some noise within cluster (~5 mile radius)
        lat = cluster[0] + random.gauss(0, 0.03)
        lon = cluster[1] + random.gauss(0, 0.03)

        # Pick industry and revenue
        industry, min_rev, max_rev = random.choice(INDUSTRIES)
        # Log-normal distribution for revenue (more small accounts, few large)
        revenue = int(random.lognormvariate(math.log(min_rev * 2), 0.5))
        revenue = max(min_rev, min(max_rev, revenue))

        # Generate company name
        prefixes = ["Acme", "Summit", "Peak", "Metro", "Prime", "Core", "Nova", "Apex"]
        suffixes = ["Solutions", "Corp", "Inc", "Group", "Systems", "Labs", "Co", "Tech"]
        name = f"{random.choice(prefixes)} {industry.split()[0]} {random.choice(suffixes)}"

        accounts.append({
            "id": i,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "revenue": revenue,
            "industry": industry,
            "name": name,
        })

    return accounts


def get_accounts() -> list[dict]:
    """Get the standard 25-account dataset used for evaluation."""
    return generate_accounts(25, SEED)


if __name__ == "__main__":
    import json
    accounts = get_accounts()
    print(json.dumps(accounts, indent=2))

    # Summary stats
    total_rev = sum(a["revenue"] for a in accounts)
    print(f"\n--- Summary ---")
    print(f"Total accounts: {len(accounts)}")
    print(f"Total revenue: ${total_rev:,}")
    print(f"Avg revenue: ${total_rev // len(accounts):,}")

    by_industry = {}
    for a in accounts:
        by_industry[a["industry"]] = by_industry.get(a["industry"], 0) + 1
    print(f"By industry: {by_industry}")
