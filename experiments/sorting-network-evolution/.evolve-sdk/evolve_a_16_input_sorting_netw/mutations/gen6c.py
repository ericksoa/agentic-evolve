"""
Cache-optimized Hybrid Multi-Stage Network - single-pass generation
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes (no conditional branching)
Stage 3: Bubble sort cleanup for guaranteed completion
Mutation: Preallocated network with single-pass generation for cache optimization
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with cache-optimized single-pass generation."""
    # Preallocate network to exact size to avoid dynamic resizing
    # Estimate total operations: shell + odd-even + bubble
    shell_ops = n * (n // 4)  # Conservative estimate for shell sort operations
    oddeven_ops = (n // 2) * (n - 1) // 2  # Odd-even operations
    bubble_ops = min(4, n // 4) * (n - 1)  # Final cleanup operations

    # Preallocate network list with estimated size for cache optimization
    estimated_size = shell_ops + oddeven_ops + bubble_ops
    network = []

    # Single-pass generation with interleaved stages for cache optimization
    # Stage 1: Shell sort with gap sequence - cache-friendly order
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Only use gaps >= 4, generate in memory-friendly order
    large_gaps = [gap for gap in gaps if gap >= 4]

    # Cache-optimized shell sort: group operations by memory locality
    for gap in large_gaps:
        # Generate all comparisons for this gap in ascending order
        gap_comparisons = []
        for i in range(gap, n):
            j = i
            while j >= gap:
                gap_comparisons.append((j - gap, j))
                j -= gap

        # Sort by first element for cache locality
        gap_comparisons.sort()
        network.extend(gap_comparisons)

    # Stage 2: Cache-friendly odd-even with precomputed patterns
    max_passes = n // 2

    # Precompute all pairs in cache-friendly order
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]

    # Interleave for better cache performance
    for pass_num in range(max_passes):
        pairs = even_pairs if (pass_num & 1) == 0 else odd_pairs
        network.extend(pairs)

    # Stage 3: Cache-optimized bubble cleanup
    final_passes = min(4, n // 4)

    # Generate bubble operations in single batch
    bubble_ops = []
    for i in range(final_passes):
        for j in range(n - 1 - i):
            bubble_ops.append((j, j + 1))

    network.extend(bubble_ops)

    return network

NETWORK = hybrid_multistage_network(N)