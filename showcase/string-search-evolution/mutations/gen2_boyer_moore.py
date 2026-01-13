"""
Boyer-Moore Algorithm
=====================

Searches from right-to-left within the pattern window.
Uses bad character heuristic to skip large portions of text.

Time complexity: O(n/m) best case (sublinear!), O(nm) worst case
Space complexity: O(k) where k = alphabet size

This is often the fastest practical algorithm for most real-world text.
Mutation type: algorithm_swap (complete algorithm change)
"""


def build_bad_char_table(pattern: str) -> dict[str, int]:
    """Build bad character skip table."""
    table = {}
    m = len(pattern)
    for i in range(m - 1):
        table[pattern[i]] = m - 1 - i
    return table


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using Boyer-Moore."""
    if not pattern or not text:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    bad_char = build_bad_char_table(pattern)
    results = []

    i = 0
    while i <= n - m:
        j = m - 1  # Start from end of pattern

        # Match from right to left
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            # Full match found
            results.append(i)
            i += 1  # Move by 1 to find overlapping matches
        else:
            # Use bad character rule
            char = text[i + j]
            skip = bad_char.get(char, m)
            i += max(1, skip - (m - 1 - j))

    return results


if __name__ == "__main__":
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Found at: {search(text, pattern)}")
