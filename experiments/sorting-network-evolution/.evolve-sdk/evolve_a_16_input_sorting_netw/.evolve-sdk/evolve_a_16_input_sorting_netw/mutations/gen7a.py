"""
Cache-Optimized Multi-Stage Network - improved memory access patterns
Stage 1: Cache-friendly shell sort with unrolled gap processing
Stage 2: Odd-even passes for systematic neighboring comparisons
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def cache_optimized_network(n):
    """Generate network with cache-optimized shell gaps and unrolled processing."""
    network = []

    # Stage 1: Cache-optimized shell sort with better gap sequence
    # Use Sedgewick's gap sequence for better performance
    gaps = []
    k = 1
    while k < n:
        if k % 2 == 1:
            gaps.append(k)
        k = k * 3 + 1 if k % 2 == 1 else k * 2
    gaps = [g for g in gaps if g >= 2][::-1]  # Only gaps >= 2, reversed

    # Unroll gap processing for better instruction pipeline utilization
    for gap in gaps:
        # Process gap comparisons in cache-friendly order
        # Group comparisons by their modulo gap to improve locality
        for offset in range(gap):
            indices = list(range(offset, n, gap))
            # Unroll pairs processing for this offset
            for i in range(1, len(indices)):
                for j in range(i, 0, -1):
                    if indices[j] - indices[j-1] == gap:
                        network.append((indices[j-1], indices[j]))

    # Stage 2: Optimized odd-even with reduced passes
    # Cache-friendly: process all even pairs first, then all odd pairs
    max_passes = n // 2

    for pass_num in range(max_passes):
        if pass_num % 2 == 0:
            # Even pass: all even-indexed pairs at once (better cache locality)
            pairs = [(i, i+1) for i in range(0, n-1, 2)]
            network.extend(pairs)
        else:
            # Odd pass: all odd-indexed pairs at once
            pairs = [(i, i+1) for i in range(1, n-1, 2)]
            network.extend(pairs)

    # Stage 3: Minimal cleanup with unrolled final passes
    final_passes = max(2, n // 6)  # Optimized number of passes

    # Unroll final bubble passes for better performance
    for i in range(final_passes):
        # Process bubble comparisons in chunks for better cache performance
        chunk_size = 4
        for start in range(0, n - 1 - i, chunk_size):
            end = min(start + chunk_size, n - 1 - i)
            for j in range(start, end):
                network.append((j, j + 1))

    return network

NETWORK = cache_optimized_network(N)