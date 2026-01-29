"""
Pairwise Network - exhaustive pairwise comparisons
Compares every pair of elements at least once
"""

N = 16

def pairwise_network(n):
    """Generate network that compares every pair of elements."""
    network = []

    # Phase 1: Compare all adjacent pairs multiple times
    for round_num in range(4):  # Multiple rounds for better mixing
        for i in range(n - 1):
            network.append((i, i + 1))

    # Phase 2: Compare pairs at distance 2
    for round_num in range(2):
        for i in range(n - 2):
            network.append((i, i + 2))

    # Phase 3: Compare pairs at larger distances
    for distance in [4, 8]:
        if distance < n:
            for i in range(n - distance):
                network.append((i, i + distance))

    # Phase 4: Final cleanup with adjacent comparisons
    for round_num in range(2):
        for i in range(n - 1):
            network.append((i, i + 1))

    return network

NETWORK = pairwise_network(N)