"""
Cache-Optimized Multi-Stage Network - memory access pattern optimization
Stage 1: Shell sort gaps optimized for cache locality
Stage 2: Unrolled odd-even passes with improved memory stride
Stage 3: Bubble sort cleanup with cache-friendly traversal
"""

N = 16

def cache_optimized_network(n):
    """Generate network with optimized memory access patterns for cache efficiency."""
    network = []

    # Stage 1: Cache-optimized shell sort with memory-friendly gap sequence
    # Use power-of-2 gaps to align with cache line boundaries
    gaps = []
    gap = n // 2
    while gap >= 2:
        gaps.append(gap)
        gap //= 2  # Powers of 2 for better cache alignment

    # Add gap=1 for final shell pass
    gaps.append(1)

    for gap in gaps:
        # Memory-locality optimized: process consecutive blocks
        for start in range(0, gap):
            for i in range(start + gap, n, gap):
                j = i
                while j >= gap:
                    network.append((j - gap, j))
                    j -= gap

    # Stage 2: Cache-friendly odd-even with loop unrolling
    # Unroll by factor of 2 to reduce loop overhead and improve cache usage
    max_passes = n // 3  # Further reduced due to better shell pre-sorting

    for pass_num in range(max_passes):
        # Unrolled even/odd pattern for better instruction pipeline
        if (pass_num & 1) == 0:
            # Even pass: unroll by pairs where possible
            for i in range(0, n-3, 4):
                network.append((i, i+1))
                network.append((i+2, i+3))
            # Handle remainder
            for i in range((n-1)//4*4, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: unroll by pairs where possible
            for i in range(1, n-3, 4):
                network.append((i, i+1))
                network.append((i+2, i+3))
            # Handle remainder
            for i in range(1 + (n-3)//4*4, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Cache-friendly bubble sort with forward-backward optimization
    # Alternate direction to improve cache locality
    final_passes = 3  # Reduced further due to improved pre-sorting

    for i in range(final_passes):
        if i & 1 == 0:
            # Forward pass
            for j in range(n - 1 - i):
                network.append((j, j + 1))
        else:
            # Backward pass for better cache reuse
            for j in range(n - 2 - i, -1, -1):
                network.append((j, j + 1))

    return network

NETWORK = cache_optimized_network(N)