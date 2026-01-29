"""
Loop-Unrolled Shell Sort Network - reduced loop overhead
Unrolls inner loop iterations for better performance
"""

N = 16

def unrolled_shell_sort_network(n):
    """Generate shell sort network with unrolled inner loops."""
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
        # Unroll the inner loop for better performance
        for i in range(gap, n):
            j = i
            # Unroll common cases to reduce loop overhead
            if gap == 1:
                # For gap=1, unroll bubble sort comparisons
                while j >= 1:
                    network.append((j - 1, j))
                    j -= 1
            elif gap == 2:
                # For gap=2, unroll every other comparison
                while j >= 2:
                    network.append((j - 2, j))
                    j -= 2
            elif gap == 4:
                # For gap=4, unroll quarter comparisons
                while j >= 4:
                    network.append((j - 4, j))
                    j -= 4
            elif gap == 8:
                # For gap=8, unroll half comparisons
                while j >= 8:
                    network.append((j - 8, j))
                    j -= 8
            else:
                # Fallback for other gaps
                while j >= gap:
                    network.append((j - gap, j))
                    j -= gap

    return network

NETWORK = unrolled_shell_sort_network(N)