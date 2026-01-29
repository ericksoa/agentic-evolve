"""
Hybrid Merge-Insert Network
Combines mergesort structure with insertion sort optimizations
"""

N = 16

def hybrid_merge_insert_network(n):
    """Generate hybrid network combining merge and insertion strategies."""
    network = []

    # First phase: parallel 2-element sorts
    for i in range(0, n, 2):
        if i + 1 < n:
            network.append((i, i + 1))

    # Second phase: merge pairs into 4-element sorted groups
    for i in range(0, n, 4):
        if i + 3 < n:
            # Merge (i,i+1) with (i+2,i+3)
            network.append((i+1, i+2))
            network.append((i, i+3))
            network.append((i+1, i+2))

    # Third phase: merge 4-element groups into 8-element groups
    for i in range(0, n, 8):
        if i + 7 < n:
            # More complex merge pattern for 8 elements
            network.append((i+3, i+4))
            network.append((i+1, i+6))
            network.append((i+2, i+5))
            network.append((i+1, i+2))
            network.append((i+5, i+6))
            network.append((i+3, i+4))

    # Final phase: merge into complete sort
    if n >= 8:
        # Merge two 8-element groups
        network.append((7, 8))
        network.append((3, 12))
        network.append((5, 10))
        network.append((6, 9))
        network.append((4, 11))
        network.append((1, 14))
        network.append((2, 13))
        network.append((1, 2))
        network.append((13, 14))
        network.append((3, 4))
        network.append((11, 12))
        network.append((5, 6))
        network.append((9, 10))
        network.append((7, 8))

    return network

NETWORK = hybrid_merge_insert_network(N)