"""
Shell Sort Network - gap-based comparisons
Adapts shell sort to fixed network with decreasing gap sizes
"""

N = 16

def shell_sort_network(n):
    """Generate shell sort inspired network with predefined gaps."""
    network = []

    # Use Knuth's gap sequence: 1, 4, 13, 40, ...
    # For n=16, we use gaps: 8, 4, 2, 1
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    for gap in gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    return network

NETWORK = shell_sort_network(N)