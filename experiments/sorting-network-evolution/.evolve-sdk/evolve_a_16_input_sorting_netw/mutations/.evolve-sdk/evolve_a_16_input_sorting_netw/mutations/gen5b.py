"""
Lookup Table Optimized Multi-Stage Network
Pre-computed comparison patterns with lookup tables for performance
Stage 1: Pre-cached Shell sort gaps
Stage 2: Lookup table for odd-even patterns
Stage 3: Minimal final cleanup with cached pairs
"""

N = 16

# Pre-compute and cache all comparison patterns at module level
_SHELL_GAPS_CACHE = None
_EVEN_PAIRS_CACHE = None
_ODD_PAIRS_CACHE = None
_FINAL_PAIRS_CACHE = None

def _initialize_lookup_tables():
    """Initialize all lookup tables once at module load."""
    global _SHELL_GAPS_CACHE, _EVEN_PAIRS_CACHE, _ODD_PAIRS_CACHE, _FINAL_PAIRS_CACHE

    # Cache shell sort gaps and their comparison pairs
    gaps = []
    k = 1
    while k < N:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    large_gaps = [gap for gap in gaps if gap >= 4]
    shell_pairs = []
    for gap in large_gaps:
        for i in range(gap, N):
            j = i
            while j >= gap:
                shell_pairs.append((j - gap, j))
                j -= gap
    _SHELL_GAPS_CACHE = shell_pairs

    # Cache even and odd comparison pairs
    _EVEN_PAIRS_CACHE = [(i, i+1) for i in range(0, N-1, 2)]
    _ODD_PAIRS_CACHE = [(i, i+1) for i in range(1, N-1, 2)]

    # Cache final cleanup pairs (reduced to minimize redundancy)
    final_pairs = []
    final_passes = min(3, N // 4)  # Reduced from 4 passes
    for i in range(final_passes):
        for j in range(N - 1 - i):
            final_pairs.append((j, j + 1))
    _FINAL_PAIRS_CACHE = final_pairs

def hybrid_lookup_network(n):
    """Generate network using pre-computed lookup tables."""
    # Initialize lookup tables if not already done
    if _SHELL_GAPS_CACHE is None:
        _initialize_lookup_tables()

    network = []

    # Stage 1: Use cached shell sort pairs
    network.extend(_SHELL_GAPS_CACHE)

    # Stage 2: Use cached odd-even patterns with optimized lookup
    max_passes = n // 2

    # Pre-build pattern lookup (even more efficient than runtime checking)
    pattern_lookup = []
    for pass_num in range(max_passes):
        if pass_num & 1 == 0:
            pattern_lookup.extend(_EVEN_PAIRS_CACHE)
        else:
            pattern_lookup.extend(_ODD_PAIRS_CACHE)

    network.extend(pattern_lookup)

    # Stage 3: Use cached final cleanup pairs
    network.extend(_FINAL_PAIRS_CACHE)

    return network

# Initialize lookup tables at module load time
_initialize_lookup_tables()

NETWORK = hybrid_lookup_network(N)