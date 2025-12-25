/**
 * String Search Seed Algorithm
 *
 * This is the starting point for evolution.
 * We use a decent but not optimal algorithm to give evolution room to improve.
 *
 * Using simplified Boyer-Moore (Horspool) as seed because:
 * 1. It's reasonably fast (gives a good baseline)
 * 2. It has clear optimization opportunities
 * 3. It can be improved in multiple directions
 */

/**
 * Find all occurrences of pattern in text
 * Returns array of starting indices
 */
export function search(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  // Handle edge cases
  if (m === 0) return results;
  if (m > n) return results;

  // Build bad character shift table
  // For each character, store how far we can shift when mismatch occurs
  const shift: Map<string, number> = new Map();

  // Default shift is pattern length (character not in pattern)
  // For characters in pattern, shift is distance from end
  for (let i = 0; i < m - 1; i++) {
    shift.set(pattern[i], m - 1 - i);
  }

  // Search
  let i = 0;
  while (i <= n - m) {
    // Compare pattern right-to-left
    let j = m - 1;
    while (j >= 0 && text[i + j] === pattern[j]) {
      j--;
    }

    if (j < 0) {
      // Found a match
      results.push(i);
      // Shift by 1 to find overlapping matches
      i += 1;
    } else {
      // Mismatch - use bad character rule
      const lastChar = text[i + m - 1];
      const shiftAmount = shift.get(lastChar) ?? m;
      i += shiftAmount;
    }
  }

  return results;
}

// Default export
export default search;
