"""
Hybrid Multi-Stage Network - Ultimate Optimization Crossover
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Highly optimized odd-even passes with minimal overhead
Stage 3: Bubble sort cleanup for guaranteed completion

Combines best aspects from gen1x, gen2a, and gen3a:
- Efficient gap computation from all parents
- Bitwise operations for performance (gen2a/gen3a)
- Precomputed ranges for minimal overhead (gen3a)
- Optimized loop structures from all parents
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with ultimate optimization combining all parent strengths."""
    network = []

    # Stage 1: Shell sort with gap sequence for initial ordering
    # Optimized gap computation (common to all parents, kept as-is)
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
    # Combines best aspects: precomputed ranges (gen3a) + bitwise ops (gen2a/gen3a)
    max_passes = n // 2

    # Precompute ranges once for maximum efficiency (from gen3a)
    even_range = range(0, n-1, 2)
    odd_range = range(1, n-1, 2)

    # Store ranges in tuple for fast indexing (optimization beyond parents)
    range_patterns = (even_range, odd_range)

    for pass_num in range(max_passes):
        # Ultra-fast pattern selection using bitwise AND and tuple indexing
        # Combines gen2a/gen3a bitwise optimization with tuple lookup
        pattern = range_patterns[pass_num & 1]
        for i in pattern:
            network.append((i, i+1))

    # Stage 3: Final bubble sort cleanup (optimized from all parents)
    # Keep the proven reduced scope approach
    final_passes = min(4, n // 4)

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)