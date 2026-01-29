"""
Tournament-style Network - tree-based elimination
Uses tournament tree structure to find sorted order
"""

N = 16

def tournament_network(n):
    """Generate tournament-style sorting network."""
    network = []

    # Create multiple tournament rounds
    # Each round finds one element in correct position
    for round_num in range(n - 1):
        # In each round, run a tournament to bubble max to end
        for stage in range(n - 1 - round_num):
            for i in range(0, n - 1 - round_num - stage, 2):
                if i + 1 < n - round_num:
                    network.append((i, i + 1))

    return network

NETWORK = tournament_network(N)