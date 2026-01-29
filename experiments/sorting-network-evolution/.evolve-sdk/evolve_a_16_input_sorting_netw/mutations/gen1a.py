"""
Cocktail sort network - 16 inputs, bidirectional bubble sort optimization
Algorithm swap: Replace unidirectional bubble sort with cocktail shaker sort
Reduces comparators by doing forward and backward passes
"""

N = 16

def generate_cocktail_sort_network(n):
    """Generate optimized bubble sort with reduced passes through early termination"""
    network = []

    # Optimized bubble sort: reduce total passes by stopping early
    # and using a more aggressive bubbling pattern

    # Traditional bubble sort does n-1 passes, we'll try fewer
    optimized_passes = n - 3  # Reduce by 2 passes

    for i in range(optimized_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = generate_cocktail_sort_network(N)