"""
Bitonic sort network mutation - 16 inputs, ~80 comparators
Algorithm family change from bubble sort to bitonic sort for better performance
"""

N = 16

def generate_bitonic_sort_network(n):
    """Generate bitonic sort network - fewer comparators than bubble sort"""
    network = []

    def bitonic_merge(low, cnt, up):
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
        if cnt > 1:
            k = cnt // 2
            bitonic_sort(low, k, True)
            bitonic_sort(low + k, k, False)
            bitonic_merge(low, cnt, up)

    bitonic_sort(0, n, True)
    return network

NETWORK = generate_bitonic_sort_network(N)