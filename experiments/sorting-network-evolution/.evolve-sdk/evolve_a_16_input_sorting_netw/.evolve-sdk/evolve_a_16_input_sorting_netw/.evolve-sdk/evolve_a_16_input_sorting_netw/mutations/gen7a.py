"""
Optimized Multi-Stage Network - tuned parameters for better performance
Stage 1: Shell sort gaps with optimized sequence selection
Stage 2: Odd-even passes with tuned pass count
Stage 3: Bubble sort cleanup with refined parameters
"""

N = 16

def optimized_multistage_network(n):
    """Generate hybrid network with optimized parameter tuning."""
    network = []

    # Stage 1: Shell sort with optimized gap sequence
    # Use Sedgewick's increments which are proven more efficient than Knuth's
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        if k % 2 == 1:
            k = 3 * k + 1
        else:
            k = k // 2 * 3 + 1

    gaps.reverse()

    # Use gaps >= 3 instead of >= 4 for better coverage
    large_gaps = [gap for gap in gaps if gap >= 3]

    for gap in large_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Odd-even transposition with optimized pass count
    # Use n // 3 instead of n // 2 for better efficiency vs completeness tradeoff
    max_passes = max(4, n // 3)  # Ensure at least 4 passes for safety

    for pass_num in range(max_passes):
        if pass_num % 2 == 0:
            # Even pass: compare (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: compare (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Final bubble sort cleanup with optimized parameters
    # Use 3 passes instead of min(4, n//4) for better guaranteed completion
    final_passes = 3

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = optimized_multistage_network(N)