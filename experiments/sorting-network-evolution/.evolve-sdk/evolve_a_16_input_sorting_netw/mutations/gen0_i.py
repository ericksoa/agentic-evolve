"""
Comb Sort Network - decreasing gap network
Adapts comb sort's gap-shrinking approach to networks
"""

N = 16

def comb_sort_network(n):
    """Generate comb sort inspired network with shrinking gaps."""
    network = []

    # Start with large gap and shrink by factor of 1.3
    gap = n
    shrink = 1.3

    while gap > 1:
        gap = max(1, int(gap / shrink))

        # Compare all elements at current gap distance
        for i in range(n - gap):
            network.append((i, i + gap))

    # Final pass with gap = 1 (bubble sort cleanup)
    for i in range(n - 1):
        network.append((i, i + 1))

    return network

NETWORK = comb_sort_network(N)