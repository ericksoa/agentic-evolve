"""
Bubble sort baseline - 16 inputs, ~120 comparators
Simple O(n^2) approach as evolutionary starting point
"""

N = 16

def generate_bubble_sort_network(n):
    """Generate bubble sort network - many comparators but guaranteed correct"""
    network = []
    for i in range(n):
        for j in range(n - 1 - i):
            network.append((j, j + 1))
    return network

NETWORK = generate_bubble_sort_network(N)