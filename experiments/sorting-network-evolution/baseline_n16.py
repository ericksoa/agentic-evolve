"""
Baseline 16-input sorting network using Batcher's odd-even mergesort.

This is a larger challenge for evolution. The known bounds are:
- Best known comparators: 60 (optimal unknown, lower bound is 53)
- Best known depth: 10

Odd-even mergesort for n=16 uses 63 comparators - close to best known!
Evolution may find ways to reduce this further.
"""

# Number of inputs to sort
N = 16

def odd_even_merge_sort_network(n):
    """Generate Batcher's odd-even mergesort network for n elements (n must be power of 2)."""
    network = []

    def odd_even_merge(lo, hi, r):
        step = r * 2
        if step < hi - lo:
            odd_even_merge(lo, hi, step)
            odd_even_merge(lo + r, hi, step)
            for i in range(lo + r, hi - r, step):
                network.append((i, i + r))
        else:
            network.append((lo, lo + r))

    def odd_even_merge_sort(lo, hi):
        if hi - lo >= 1:
            mid = lo + (hi - lo) // 2
            odd_even_merge_sort(lo, mid)
            odd_even_merge_sort(mid + 1, hi)
            odd_even_merge(lo, hi, 1)

    odd_even_merge_sort(0, n - 1)

    # Normalize: ensure i < j for all comparators
    normalized = []
    for i, j in network:
        if i < j:
            normalized.append((i, j))
        else:
            normalized.append((j, i))

    return normalized

# Generate the network
NETWORK = odd_even_merge_sort_network(N)

print(f"Generated {len(NETWORK)} comparators for n={N}")
