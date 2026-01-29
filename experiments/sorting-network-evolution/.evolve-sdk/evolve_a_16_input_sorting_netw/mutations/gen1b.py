"""
Bitonic Sort Network - divide and conquer approach
Optimized for 16 elements with reduced comparisons
"""

N = 16

def bitonic_merge_network(start, cnt, up):
    """Generate comparisons for bitonic merge."""
    network = []
    if cnt > 1:
        k = cnt // 2
        for i in range(start, start + k):
            if up:
                network.append((i, i + k))
            else:
                network.append((i + k, i))
        network.extend(bitonic_merge_network(start, k, up))
        network.extend(bitonic_merge_network(start + k, k, up))
    return network

def bitonic_sort_network(start, cnt, up):
    """Generate comparisons for bitonic sort."""
    network = []
    if cnt > 1:
        k = cnt // 2
        # Sort first half in ascending order
        network.extend(bitonic_sort_network(start, k, True))
        # Sort second half in descending order
        network.extend(bitonic_sort_network(start + k, k, False))
        # Merge the whole sequence
        network.extend(bitonic_merge_network(start, cnt, up))
    return network

def generate_network(n):
    """Generate optimized bitonic sorting network for n elements."""
    return bitonic_sort_network(0, n, True)

NETWORK = generate_network(N)