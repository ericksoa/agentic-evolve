"""
Rabin-Karp Algorithm
====================

Uses rolling hash to quickly compare pattern with text windows.
Only does character-by-character comparison when hashes match.

Time complexity: O(n + m) average, O(nm) worst case (many hash collisions)
Space complexity: O(1)

Excellent for multiple pattern search. Hash computation is constant time per position.
Mutation type: algorithm_swap (complete algorithm change)
"""

# Large prime for modular arithmetic
PRIME = 101
BASE = 256


def search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using Rabin-Karp."""
    if not pattern or not text:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    results = []

    # Compute hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = pow(BASE, m - 1, PRIME)  # BASE^(m-1) mod PRIME

    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % PRIME
        window_hash = (BASE * window_hash + ord(text[i])) % PRIME

    # Slide the window
    for i in range(n - m + 1):
        # Check if hashes match
        if pattern_hash == window_hash:
            # Verify character by character (avoid false positives)
            if text[i : i + m] == pattern:
                results.append(i)

        # Compute hash for next window (rolling hash)
        if i < n - m:
            window_hash = (BASE * (window_hash - ord(text[i]) * h) + ord(text[i + m])) % PRIME
            if window_hash < 0:
                window_hash += PRIME

    return results


if __name__ == "__main__":
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print(f"Found at: {search(text, pattern)}")
