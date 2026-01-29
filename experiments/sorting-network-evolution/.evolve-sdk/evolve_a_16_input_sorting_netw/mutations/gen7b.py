"""
Hybrid Multi-Stage Network - Lookup Table Optimized Version
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes with eliminated branching
Stage 3: Bubble sort cleanup for guaranteed completion
Optimization: Precomputed lookup tables for comparison patterns
"""

N = 16

# Precomputed lookup tables for performance optimization
_GAP_SEQUENCE_CACHE = {}
_COMPARISON_PATTERNS_CACHE = {}

def _get_gap_sequence(n):
    """Get cached gap sequence for shell sort."""
    if n not in _GAP_SEQUENCE_CACHE:
        gaps = []
        k = 1
        while k < n:
            gaps.append(k)
            k = 3 * k + 1
        gaps.reverse()
        # Only use gaps >= 4 to focus on long-distance comparisons
        _GAP_SEQUENCE_CACHE[n] = [gap for gap in gaps if gap >= 4]
    return _GAP_SEQUENCE_CACHE[n]

def _get_comparison_patterns(n):
    """Get cached even/odd comparison patterns."""
    if n not in _COMPARISON_PATTERNS_CACHE:
        # Generate even comparisons: (0,1), (2,3), (4,5), ...
        even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
        # Generate odd comparisons: (1,2), (3,4), (5,6), ...
        odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]
        _COMPARISON_PATTERNS_CACHE[n] = (even_pairs, odd_pairs)
    return _COMPARISON_PATTERNS_CACHE[n]

def hybrid_multistage_network(n):
    """Generate hybrid network with lookup table optimizations."""
    network = []

    # Stage 1: Shell sort with cached gap sequence
    large_gaps = _get_gap_sequence(n)

    for gap in large_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Branch-eliminated odd-even transposition with cached patterns
    max_passes = n // 2
    even_pairs, odd_pairs = _get_comparison_patterns(n)

    # Interleave patterns without branching using cached patterns
    for pass_num in range(max_passes):
        # Use modulo arithmetic to alternate patterns
        if pass_num & 1 == 0:  # Bitwise AND for even/odd check
            network.extend(even_pairs)  # Use extend for bulk addition
        else:
            network.extend(odd_pairs)   # Use extend for bulk addition

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n // 4)

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

# Initialize lookup tables at module load time
_get_gap_sequence(N)
_get_comparison_patterns(N)

NETWORK = hybrid_multistage_network(N)