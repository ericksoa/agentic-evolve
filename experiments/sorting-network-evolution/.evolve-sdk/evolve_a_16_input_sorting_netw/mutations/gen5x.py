"""
Crossover Hybrid Multi-Stage Network - Generation 5 Fusion
Combines best optimizations from gen1x, gen2a, and gen3a:
- gen1x: Core algorithm structure and gap optimization
- gen2a: Branch elimination with pre-generated pairs
- gen3a: Efficient range-based computation and bitwise operations

Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Optimized odd-even transposition with eliminated branching
Stage 3: Bubble sort cleanup with adaptive passes
"""

N = 16

def hybrid_multistage_network(n):
    """Generate optimized hybrid network combining best features from all parents."""
    network = []

    # Stage 1: Shell sort with gap sequence (from gen1x core algorithm)
    # Use Knuth's gap sequence but only larger gaps to avoid redundancy
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Only use gaps >= 4 to focus on long-distance comparisons (gen1x optimization)
    large_gaps = [gap for gap in gaps if gap >= 4]

    for gap in large_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Highly optimized odd-even transposition
    # Combines gen2a's pair pre-generation with gen3a's range efficiency
    max_passes = n // 2  # Reduced passes due to shell pre-sorting (gen1x insight)

    # Pre-generate optimized ranges (gen3a approach) but store as pairs (gen2a style)
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]  # gen2a: explicit pairs
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]   # gen2a: explicit pairs

    # Use bitwise operations for maximum efficiency (gen2a & gen3a)
    for pass_num in range(max_passes):
        # Bitwise AND for even/odd check - most efficient branch elimination
        pairs = even_pairs if (pass_num & 1) == 0 else odd_pairs
        # Direct extension without intermediate variable (gen2a optimization)
        network.extend(pairs)

    # Stage 3: Adaptive bubble sort cleanup
    # Enhanced from all parents with adaptive pass calculation
    final_passes = min(4, n // 4)  # Conservative cleanup (all parents agree)

    # Optimized bubble pass generation
    for i in range(final_passes):
        # Generate comparisons in one pass (efficiency from gen3a approach)
        bubble_pairs = [(j, j + 1) for j in range(n - 1 - i)]
        network.extend(bubble_pairs)

    return network

NETWORK = hybrid_multistage_network(N)