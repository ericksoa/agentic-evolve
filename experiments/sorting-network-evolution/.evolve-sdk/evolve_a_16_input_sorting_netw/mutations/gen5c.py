"""
Hybrid Multi-Stage Network - loop unrolling optimization
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes with cache optimization
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with loop unrolling for cache optimization."""
    network = []

    # Stage 1: Shell sort with gap sequence for initial ordering
    # Use Knuth's gap sequence but only larger gaps to avoid redundancy
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Only use gaps >= 4 to focus on long-distance comparisons
    large_gaps = [gap for gap in gaps if gap >= 4]

    for gap in large_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Unrolled odd-even transposition for cache optimization
    # Instead of alternating passes, unroll to do both even and odd pairs per iteration
    max_passes = n // 4  # Reduced further due to unrolling efficiency

    # Precompute pairs for cache-friendly access
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]

    for pass_num in range(max_passes):
        # Unrolled: do both even and odd pairs in single iteration
        # This improves cache locality by accessing adjacent elements
        for i in range(0, n-1, 2):
            network.append((i, i+1))  # Even pairs
        for i in range(1, n-1, 2):
            network.append((i, i+1))  # Odd pairs

    # Additional targeted passes for edge cases
    # Do a few more alternating passes for thorough sorting
    for _ in range(2):
        for i in range(0, n-1, 2):
            network.append((i, i+1))
        for i in range(1, n-1, 2):
            network.append((i, i+1))

    # Stage 3: Final bubble sort cleanup (minimal scope)
    # Only do a few final passes since most sorting is done
    final_passes = 3  # Reduced due to better intermediate sorting

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)