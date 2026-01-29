"""
Hybrid Multi-Stage Network - Performance Optimized Crossover
Combines the best features from gen1x, gen2a, and gen3a:
- Shell sort gaps for initial long-distance ordering
- Ultra-optimized odd-even passes with minimal branching and cache optimization
- Efficient bubble sort cleanup
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network combining best optimizations from all parents."""
    network = []

    # Stage 1: Shell sort with gap sequence for initial ordering
    # Inherited from all parents - proven effective approach
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

    # Stage 2: Ultra-optimized odd-even transposition
    # Combines gen3a's efficient range approach with gen2a's precomputation strategy
    max_passes = n // 2  # Reduced passes due to shell pre-sorting

    # Precompute ranges for maximum efficiency (from gen3a)
    even_range = list(range(0, n-1, 2))
    odd_range = list(range(1, n-1, 2))

    # Store ranges in a tuple for even faster access pattern
    range_patterns = (even_range, odd_range)

    # Unrolled optimization: process pairs in batches when possible
    for pass_num in range(max_passes):
        # Use bitwise AND for fastest even/odd check (from gen2a/gen3a)
        current_range = range_patterns[pass_num & 1]

        # Batch append for better performance (new optimization)
        batch_pairs = [(i, i+1) for i in current_range]
        network.extend(batch_pairs)

    # Stage 3: Optimized bubble sort cleanup
    # Maintains the proven reduced scope from all parents
    final_passes = min(4, n // 4)  # Conservative cleanup passes

    for i in range(final_passes):
        # Pre-calculate range for better loop optimization
        cleanup_range = range(n - 1 - i)
        cleanup_pairs = [(j, j + 1) for j in cleanup_range]
        network.extend(cleanup_pairs)

    return network

NETWORK = hybrid_multistage_network(N)