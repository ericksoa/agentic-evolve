"""
Optimized Hybrid Multi-Stage Network - reduced gap threshold and increased final passes
Stage 1: Shell sort gaps for initial long-distance ordering (reduced gap threshold)
Stage 2: Odd-even passes for systematic neighboring comparisons
Stage 3: Enhanced bubble sort cleanup for guaranteed completion
"""

N = 16

def optimized_hybrid_multistage_network(n):
    """Generate optimized hybrid network with tuned parameters for better performance."""
    network = []

    # Stage 1: Shell sort with gap sequence for initial ordering
    # Use Knuth's gap sequence but with LOWER threshold for more thorough initial sorting
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Use gaps >= 2 instead of >= 4 for more comprehensive initial sorting
    medium_gaps = [gap for gap in gaps if gap >= 2]

    for gap in medium_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Odd-even transposition for systematic neighboring comparisons
    # Slightly increased passes for better intermediate sorting
    max_passes = (n // 2) + 1  # Increased from n//2 to (n//2)+1

    for pass_num in range(max_passes):
        if pass_num % 2 == 0:
            # Even pass: compare (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: compare (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Enhanced final bubble sort cleanup
    # Slightly increased final passes for guaranteed correctness
    final_passes = min(6, n // 3)  # Increased from min(4, n//4) to min(6, n//3)

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = optimized_hybrid_multistage_network(N)