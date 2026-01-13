"""
Naive with Early Exit Optimization
===================================

Minor improvement: break on first mismatch (baseline already does this).
This represents a parameter_tweak mutation - minimal structural change.

Time complexity: O(n * m) - same as baseline
Space complexity: O(1)

This mutation type has low impact on this problem.
Mutation type: parameter_tweak
"""


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text with early exit."""
    if not pattern or not text:
        return []

    results = []
    n, m = len(text), len(pattern)

    # Cache first character for quick rejection
    first_char = pattern[0]

    for i in range(n - m + 1):
        # Quick check: first character
        if text[i] != first_char:
            continue

        # Full match check
        match = True
        for j in range(1, m):  # Start from 1 since we checked 0
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
