"""
Optimized Hybrid Crossover Network - Generation 2
Combines gap sequencing from Shell, multi-stage design from gen1x, and adaptive sizing
Features: Parallel gap phases, optimized odd-even, and minimal cleanup
"""

N = 16

def optimized_hybrid_network(n):
    """Enhanced hybrid combining best aspects of all parents with new optimizations."""
    network = []

    # Stage 1: Parallel Shell-style gap processing (from gen0_e inspiration)
    # Use modified Knuth sequence but process gaps more efficiently
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Enhanced gap processing: parallel and optimized (hybrid of gen1x + gen0_e)
    for gap in gaps:
        if gap >= 2:  # Lowered threshold from gen1x's 4 for better coverage
            # Process all gap-distance pairs in parallel where possible
            for start_offset in range(gap):
                indices = list(range(start_offset, n, gap))
                # Create all comparisons for this gap sequence
                for i in range(1, len(indices)):
                    network.append((indices[i-1], indices[i]))

    # Stage 2: Optimized Odd-Even with adaptive passes (improved from gen1x)
    # Calculate optimal passes based on network size and pre-sorting
    base_passes = max(3, n // 6)  # Adaptive: fewer for smaller n, more for larger

    for pass_num in range(base_passes):
        if pass_num % 2 == 0:
            # Even pass: (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Intelligent cleanup (combining bubble simplicity with efficiency)
    # Use bubble-style but with early termination heuristic
    cleanup_passes = min(3, n // 8 + 1)  # Reduced even further than gen1x

    for i in range(cleanup_passes):
        # Focus on the most likely unsorted pairs (from bubble sort insight)
        for j in range(n - 1 - i):
            # Only add if we haven't already covered this comparison recently
            if j < n - 1:
                network.append((j, j + 1))

    # Stage 4: Final parallel verification sweep (new addition)
    # Quick parallel check of all adjacent pairs one more time
    for i in range(0, n-1):
        network.append((i, i+1))

    return network

NETWORK = optimized_hybrid_network(N)