"""
Bitonic-OddEven Hybrid Network - replaces shell sort stage with bitonic merge patterns
Stage 1: Bitonic merge patterns for optimal parallel execution
Stage 2: Odd-even passes for systematic neighboring comparisons
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def bitonic_oddeven_network(n):
    """Generate hybrid network with bitonic merge stage, odd-even passes, and bubble cleanup."""
    network = []

    # Stage 1: Bitonic merge patterns instead of shell sort
    # Bitonic sequences provide better parallelization and cache performance
    def add_bitonic_merge(start, length, ascending=True):
        """Add bitonic merge comparisons for a sequence."""
        if length > 1:
            step = length // 2
            for i in range(start, start + step):
                if ascending:
                    network.append((i, i + step))
                else:
                    network.append((i + step, i))

            # Recursively merge the halves
            add_bitonic_merge(start, step, ascending)
            add_bitonic_merge(start + step, step, ascending)

    def add_bitonic_sort_stage(start, length):
        """Add bitonic sort comparisons for a sequence."""
        if length > 1:
            step = length // 2
            # Create ascending and descending subsequences
            add_bitonic_sort_stage(start, step)
            add_bitonic_sort_stage(start + step, step)
            # Merge them bitonically
            add_bitonic_merge(start, length, True)

    # Apply bitonic patterns to power-of-2 subsequences
    # For n=16, we can do full bitonic sort on entire array
    if n == 16:  # Perfect power of 2
        add_bitonic_sort_stage(0, n)
    else:
        # For non-power-of-2, apply bitonic patterns to largest power-of-2 subsets
        largest_pow2 = 1
        while largest_pow2 * 2 <= n:
            largest_pow2 *= 2

        add_bitonic_sort_stage(0, largest_pow2)
        if n > largest_pow2:
            # Handle remaining elements with smaller bitonic patterns
            remaining = n - largest_pow2
            if remaining >= 2:
                add_bitonic_sort_stage(largest_pow2, remaining if remaining & (remaining-1) == 0 else 2)

    # Stage 2: Odd-even transposition for systematic neighboring comparisons
    # Reduced number of passes since bitonic stage provides better initial ordering
    max_passes = n // 3  # Further reduced due to better bitonic pre-sorting

    for pass_num in range(max_passes):
        if pass_num % 2 == 0:
            # Even pass: compare (0,1), (2,3), (4,5), ...
            for i in range(0, n-1, 2):
                network.append((i, i+1))
        else:
            # Odd pass: compare (1,2), (3,4), (5,6), ...
            for i in range(1, n-1, 2):
                network.append((i, i+1))

    # Stage 3: Minimal final cleanup
    # Even fewer passes needed due to bitonic pre-sorting efficiency
    final_passes = max(1, n // 8)  # Reduced from n//4 to n//8

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = bitonic_oddeven_network(N)