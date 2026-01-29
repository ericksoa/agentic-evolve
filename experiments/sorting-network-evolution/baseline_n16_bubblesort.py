"""
Baseline 16-input sorting network using bubble sort construction.

Bubble sort network for n=16 produces 120 comparators - significantly more than
the best known (60), giving evolution 60 comparators to optimize away (50% reduction!).

Known bounds for n=16:
- Best known comparators: 60 (optimal unknown, lower bound is 53)
- Best known depth: 10
- This baseline: 120 comparators (50% reduction possible!)

This is a great starting point for evolution because:
1. It's provably correct (bubble sort always works)
2. Massive room for improvement (60 extra comparators)
3. Evolution can discover structural optimizations
"""

# Number of inputs to sort
N = 16

def bubble_sort_network(n):
    """
    Generate a bubble sort network for n elements.

    This is the simplest sorting network construction:
    - n-1 passes
    - Each pass compares adjacent elements
    - Total comparators: n*(n-1)/2 = 120 for n=16
    """
    network = []

    for i in range(n - 1):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

# Generate the network
NETWORK = bubble_sort_network(N)

# Verification
assert len(NETWORK) == N * (N - 1) // 2, f"Expected {N*(N-1)//2} comparators, got {len(NETWORK)}"
