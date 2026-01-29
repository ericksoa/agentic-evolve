"""
Hybrid Multi-Stage Network - Crossover Optimization
Combines best performance optimizations from generations 1-3:
- Bitwise operations for faster even/odd checking (gen2a/gen3a)
- Pre-computed pair tuples for direct appending (gen2a)
- Streamlined range computation (gen3a)
- Additional memory efficiency optimizations

Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Optimized odd-even passes with eliminated branching
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with maximum performance optimizations."""
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

    # Stage 2: Maximally optimized odd-even transposition
    # Combine best elements from all parents:
    # - Pre-computed tuple pairs (gen2a approach)
    # - Bitwise operations (gen2a/gen3a)
    # - Efficient list comprehensions
    max_passes = n >> 1  # Bit shift instead of division for n//2

    # Pre-compute pairs as tuples for direct appending (gen2a optimization)
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]

    # Use bitwise operations and tuple lookup for maximum efficiency
    pair_lookup = [even_pairs, odd_pairs]

    for pass_num in range(max_passes):
        # Bitwise AND with direct lookup (combines gen2a/gen3a optimizations)
        pairs = pair_lookup[pass_num & 1]
        network.extend(pairs)  # extend() is faster than repeated append()

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n >> 2)  # Bit shift for n//4

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)