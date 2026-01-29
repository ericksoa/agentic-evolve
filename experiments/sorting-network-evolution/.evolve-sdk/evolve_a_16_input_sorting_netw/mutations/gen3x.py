"""
Optimized Multi-Stage Hybrid Network - Gen 3 Crossover
Combines: Multi-stage approach + Branch elimination + Loop unrolling
Stage 1: Unrolled Shell sort gaps for initial long-distance ordering
Stage 2: Branch-eliminated odd-even passes for systematic comparisons
Stage 3: Optimized bubble sort cleanup
"""

N = 16

def optimized_hybrid_network(n):
    """Generate optimized hybrid network combining best features from all parents."""
    network = []

    # Stage 1: Unrolled Shell sort with gap sequence (from gen2c optimization)
    # Use Knuth's gap sequence but with unrolled inner loops for performance
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Only use gaps >= 4 for long-distance comparisons (from gen1x insight)
    large_gaps = [gap for gap in gaps if gap >= 4]

    for gap in large_gaps:
        # Unrolled inner loops for common gap values (from gen2c)
        for i in range(gap, n):
            j = i
            if gap == 4:
                # Unroll gap=4 comparisons for better performance
                while j >= 4:
                    network.append((j - 4, j))
                    j -= 4
            elif gap == 8:
                # Unroll gap=8 comparisons
                while j >= 8:
                    network.append((j - 8, j))
                    j -= 8
            else:
                # Fallback for other gaps
                while j >= gap:
                    network.append((j - gap, j))
                    j -= gap

    # Stage 2: Branch-eliminated odd-even transposition (from gen2a optimization)
    max_passes = n // 2  # Reduced due to shell pre-sorting

    # Pre-generate patterns to eliminate runtime branching (from gen2a)
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]

    # Use bitwise operations for efficient even/odd detection (from gen2a)
    for pass_num in range(max_passes):
        if pass_num & 1 == 0:  # Bitwise AND for even check
            for pair in even_pairs:
                network.append(pair)
        else:
            for pair in odd_pairs:
                network.append(pair)

    # Stage 3: Optimized final cleanup with unrolling (hybrid innovation)
    final_passes = min(4, n // 4)  # Reduced passes from gen1x insight

    # Unroll the final bubble passes for better performance
    for i in range(final_passes):
        # Unroll adjacent comparisons in groups for cache efficiency
        remaining = n - 1 - i
        j = 0
        # Process in groups of 4 for better cache utilization
        while j + 3 < remaining:
            network.append((j, j + 1))
            network.append((j + 1, j + 2))
            network.append((j + 2, j + 3))
            network.append((j + 3, j + 4))
            j += 4
        # Handle remaining comparisons
        while j < remaining:
            network.append((j, j + 1))
            j += 1

    return network

NETWORK = optimized_hybrid_network(N)