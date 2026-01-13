"""
Knuth-Morris-Pratt (KMP) Algorithm
==================================

Uses a failure function to skip unnecessary comparisons.
When a mismatch occurs, we know how much of the pattern still matches.

Time complexity: O(n + m) - linear!
Space complexity: O(m) for the failure table

This represents a major algorithmic improvement over naive search.
Mutation type: algorithm_swap (complete algorithm change)
"""


def compute_lps(pattern: str) -> list[int]:
    """Compute Longest Proper Prefix which is also Suffix (LPS) array."""
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using KMP."""
    if not pattern or not text:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    lps = compute_lps(pattern)
    results = []

    i = 0  # index for text
    j = 0  # index for pattern

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            results.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results


if __name__ == "__main__":
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Found at: {search(text, pattern)}")
