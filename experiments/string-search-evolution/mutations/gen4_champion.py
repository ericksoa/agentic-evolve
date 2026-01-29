"""
Optimized Boyer-Moore-Horspool with Sunday's Quick Search Enhancement
=====================================================================

Combines Boyer-Moore-Horspool simplicity with Sunday's improvement:
- Uses only the bad character heuristic (simpler, often faster)
- Looks at character AFTER the current window for skip calculation
- Optimized inner loop with early termination

Time complexity: O(n/m) best case (sublinear!)
Space complexity: O(k) where k = alphabet size

This hybrid emerged through crossover of gen2 (Boyer-Moore) and structural optimizations.
Mutation type: crossover + structural
"""


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using optimized BMH+Sunday."""
    if not pattern or not text:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Build skip table - characters not in pattern skip by m+1
    skip = {}
    for i in range(m):
        skip[pattern[i]] = m - i

    results = []
    i = 0

    # Cache pattern as bytes for faster comparison
    pattern_bytes = pattern.encode() if isinstance(pattern, str) else pattern
    text_bytes = text.encode() if isinstance(text, str) else text

    while i <= n - m:
        j = 0
        # Forward comparison (often better cache behavior)
        while j < m and text_bytes[i + j] == pattern_bytes[j]:
            j += 1

        if j == m:
            results.append(i)
            i += 1  # Allow overlapping matches
        else:
            # Sunday's enhancement: look at character after window
            if i + m < n:
                next_char = text[i + m]
                i += skip.get(next_char, m + 1)
            else:
                i += 1

    return results


if __name__ == "__main__":
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Found at: {search(text, pattern)}")
