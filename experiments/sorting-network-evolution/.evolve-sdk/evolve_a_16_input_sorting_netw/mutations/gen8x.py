"""
Crossover Hybrid Multi-Stage Network - Generation 8
Combines optimizations from gen1x, gen2a, and gen3a:
- Shell sort gaps for initial long-distance ordering
- Optimized odd-even passes with precomputed ranges and bitwise operations
- Bubble sort cleanup with adaptive pass count
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network combining best features from parents."""
    network = []

    # Stage 1: Shell sort with gap sequence (from all parents)
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

    # Stage 2: Optimized branch-eliminated odd-even transposition
    # Combines gen2a's pair lists with gen3a's range optimization
    max_passes = n // 2  # Reduced due to shell pre-sorting

    # Precompute ranges for maximum efficiency (from gen3a)
    even_indices = list(range(0, n-1, 2))
    odd_indices = list(range(1, n-1, 2))

    # Pre-generate pair lists for better cache locality (from gen2a)
    even_pairs = [(i, i+1) for i in even_indices]
    odd_pairs = [(i, i+1) for i in odd_indices]

    # Use bitwise operation for even/odd detection (from gen3a)
    for pass_num in range(max_passes):
        pairs = even_pairs if (pass_num & 1) == 0 else odd_pairs
        # Extend network efficiently
        network.extend(pairs)

    # Stage 3: Adaptive bubble sort cleanup
    # Enhanced from parents with better adaptation
    final_passes = min(4, max(2, n // 4))  # Ensure at least 2 cleanup passes

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)