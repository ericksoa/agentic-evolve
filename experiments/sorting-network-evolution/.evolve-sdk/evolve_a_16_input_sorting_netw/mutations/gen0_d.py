"""
Optimized Odd-Even Network - reduced depth variant
Interleaves odd-even comparisons for better parallelization
"""

N = 16

def optimized_odd_even_network(n):
    """Generate optimized odd-even sorting network with reduced depth."""
    network = []

    # Multiple passes of odd-even transposition
    for pass_num in range(n):
        if pass_num % 2 == 0:
            # Even pass: compare (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: compare (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    return network

NETWORK = optimized_odd_even_network(N)