"""
String Search Baseline: Naive Algorithm
========================================

Find all occurrences of a pattern in text.
This is the simplest approach - check every position.

Time complexity: O(n * m) where n = text length, m = pattern length
"""


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text. Returns list of starting indices."""
    if not pattern or not text:
        return []

    results = []
    n, m = len(text), len(pattern)

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            results.append(i)

    return results


if __name__ == "__main__":
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Found at: {search(text, pattern)}")
