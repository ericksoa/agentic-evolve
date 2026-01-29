"""
Bitonic Sorting Network - optimized for cache locality
Uses recursive bitonic merge structure for better memory access patterns
"""

N = 16

def bitonic_sorting_network(n):
    """Generate bitonic sorting network with optimized memory access patterns."""
    network = []

    def bitonic_merge(low, cnt, up):
        """Generate comparisons for bitonic merge."""
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if up:
                    network.append((i, i + k))
                else:
                    network.append((i + k, i))
            bitonic_merge(low, k, up)
            bitonic_merge(low + k, k, up)

    def bitonic_sort(low, cnt, up):
        """Generate comparisons for bitonic sort."""
        if cnt > 1:
            k = cnt // 2
            # Sort first half in ascending order
            bitonic_sort(low, k, True)
            # Sort second half in descending order
            bitonic_sort(low + k, k, False)
            # Merge the whole sequence
            bitonic_merge(low, cnt, up)

    # Generate the bitonic sorting network
    bitonic_sort(0, n, True)

    return network

NETWORK = bitonic_sorting_network(N)