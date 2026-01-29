"""
Novel Hierarchical Network - custom balanced tree approach
Uses recursive subdivision with balanced comparisons
"""

N = 16

def hierarchical_network(n):
    """Generate novel hierarchical sorting network."""
    network = []

    def sort_range(start, end):
        """Recursively sort a range using hierarchical approach."""
        length = end - start + 1
        if length <= 1:
            return
        elif length == 2:
            network.append((start, end))
            return

        # Split into thirds for more balanced tree
        third = length // 3
        mid1 = start + third - 1
        mid2 = start + 2 * third - 1

        # Recursively sort each third
        sort_range(start, mid1)
        sort_range(mid1 + 1, mid2)
        sort_range(mid2 + 1, end)

        # Merge the three sorted sections
        # First merge sections 1&2, then merge result with section 3
        merge_sections(start, mid1, mid2)
        merge_sections(start, mid2, end)

    def merge_sections(start, mid, end):
        """Merge two adjacent sorted sections."""
        # Simple merge with interleaved comparisons
        for offset in range(min(mid - start + 1, end - mid)):
            for i in range(start + offset, mid + 1):
                if i + (end - mid) <= end:
                    network.append((i, i + (end - mid)))

    sort_range(0, n - 1)
    return network

NETWORK = hierarchical_network(N)