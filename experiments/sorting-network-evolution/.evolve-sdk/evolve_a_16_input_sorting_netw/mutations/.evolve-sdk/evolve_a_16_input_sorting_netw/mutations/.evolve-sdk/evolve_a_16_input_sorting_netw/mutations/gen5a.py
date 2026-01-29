"""
Hybrid Multi-Stage Network with Loop Unrolling Optimization
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes for systematic neighboring comparisons
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with unrolled odd-even passes for better performance."""
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

    # Stage 2: Unrolled odd-even transposition for better performance
    # Manually unroll 4 passes at a time to reduce loop overhead
    max_passes = n // 2  # Reduced from n to n//2 due to shell pre-sorting

    # Process passes in groups of 4 for unrolling
    full_groups = max_passes // 4
    remaining_passes = max_passes % 4

    for group in range(full_groups):
        base_pass = group * 4

        # Unrolled pass 0 (even)
        for i in range(0, n-1, 2):
            network.append((i, i+1))

        # Unrolled pass 1 (odd)
        for i in range(1, n-1, 2):
            network.append((i, i+1))

        # Unrolled pass 2 (even)
        for i in range(0, n-1, 2):
            network.append((i, i+1))

        # Unrolled pass 3 (odd)
        for i in range(1, n-1, 2):
            network.append((i, i+1))

    # Handle remaining passes that don't fit in groups of 4
    for pass_offset in range(remaining_passes):
        pass_num = full_groups * 4 + pass_offset
        if pass_num % 2 == 0:
            # Even pass: compare (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: compare (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n // 4)  # Much fewer passes than full bubble sort

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)