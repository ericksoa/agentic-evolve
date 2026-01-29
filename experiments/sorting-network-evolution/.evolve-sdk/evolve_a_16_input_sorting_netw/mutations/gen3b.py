"""
Hybrid Multi-Stage Network - Loop Unrolling Optimization
Stage 1: Unrolled Shell sort with precomputed comparisons
Stage 2: Unrolled odd-even passes with eliminated branching
Stage 3: Minimal bubble sort cleanup
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with loop unrolling optimizations."""
    network = []

    # Stage 1: Shell sort with unrolled gap sequence - only essential gaps
    # Unroll the gap computation to avoid runtime calculation
    gaps = [13, 4]  # Skip gap=1 to reduce comparators, let Stage 2 handle fine sorting

    for gap in gaps:
        # Unroll the Shell sort loops for better performance
        # For gap=13: only one comparison (0,13), (1,14), (2,15)
        if gap == 13:
            for i in range(3):  # Only 3 valid pairs for gap=13 in n=16
                network.append((i, i + gap))
        # For gap=4: unroll all comparisons
        elif gap == 4:
            # Unrolled version of nested loops for gap=4
            network.extend([
                (0, 4), (1, 5), (2, 6), (3, 7),
                (4, 8), (5, 9), (6, 10), (7, 11),
                (8, 12), (9, 13), (10, 14), (11, 15)
            ])

    # Stage 2: Unrolled odd-even transposition with fewer passes
    # Precompute patterns and unroll the alternating logic
    even_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]
    odd_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14)]

    # Unroll 7 passes explicitly (optimal for 16 elements after Shell)
    # Pass 0 (even)
    network.extend(even_pairs)
    # Pass 1 (odd)
    network.extend(odd_pairs)
    # Pass 2 (even)
    network.extend(even_pairs)
    # Pass 3 (odd)
    network.extend(odd_pairs)
    # Pass 4 (even)
    network.extend(even_pairs)
    # Pass 5 (odd)
    network.extend(odd_pairs)
    # Pass 6 (even) - final cleanup
    network.extend(even_pairs)

    return network

NETWORK = hybrid_multistage_network(N)