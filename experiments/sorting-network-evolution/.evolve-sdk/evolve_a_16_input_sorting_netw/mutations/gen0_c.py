"""
Simple Recursive Merge Network - divide and conquer approach
Recursively merges smaller sorted subsequences
"""

N = 16

def recursive_merge_network(n):
    """Generate recursive merge sorting network."""
    network = []

    def merge_network(start1, end1, start2, end2):
        """Add comparators to merge two sorted ranges."""
        if start1 > end1 or start2 > end2:
            return

        # Simple merge: compare elements from each half
        len1 = end1 - start1 + 1
        len2 = end2 - start2 + 1

        for i in range(min(len1, len2)):
            network.append((start1 + i, start2 + i))

        # Handle remaining elements
        if len1 > len2:
            for i in range(len2, len1):
                if start1 + i <= end1:
                    for j in range(start2, end2 + 1):
                        network.append((min(start1 + i, j), max(start1 + i, j)))

    def sort_network(start, end):
        """Recursively build sorting network for range."""
        if start >= end:
            return

        if end - start == 1:
            network.append((start, end))
            return

        mid = (start + end) // 2
        sort_network(start, mid)
        sort_network(mid + 1, end)
        merge_network(start, mid, mid + 1, end)

    sort_network(0, n - 1)
    return network

NETWORK = recursive_merge_network(N)