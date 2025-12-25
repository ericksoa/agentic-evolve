/**
 * String Search Baseline Algorithms
 *
 * These are well-known, battle-tested string search algorithms.
 * Our evolved algorithm will be benchmarked against these.
 */

/**
 * Naive string search - O(n*m)
 * Simple but slow reference implementation
 */
export function naiveSearch(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  for (let i = 0; i <= n - m; i++) {
    let match = true;
    for (let j = 0; j < m; j++) {
      if (text[i + j] !== pattern[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      results.push(i);
    }
  }

  return results;
}

/**
 * Knuth-Morris-Pratt (KMP) - O(n+m)
 * Uses failure function to avoid redundant comparisons
 */
export function kmpSearch(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  if (m === 0) return results;

  // Build failure function
  const failure = buildKMPFailure(pattern);

  let i = 0; // text index
  let j = 0; // pattern index

  while (i < n) {
    if (text[i] === pattern[j]) {
      i++;
      j++;
      if (j === m) {
        results.push(i - m);
        j = failure[j - 1];
      }
    } else if (j > 0) {
      j = failure[j - 1];
    } else {
      i++;
    }
  }

  return results;
}

function buildKMPFailure(pattern: string): number[] {
  const m = pattern.length;
  const failure = new Array(m).fill(0);

  let i = 1;
  let j = 0;

  while (i < m) {
    if (pattern[i] === pattern[j]) {
      j++;
      failure[i] = j;
      i++;
    } else if (j > 0) {
      j = failure[j - 1];
    } else {
      failure[i] = 0;
      i++;
    }
  }

  return failure;
}

/**
 * Boyer-Moore - O(n/m) best case, O(n*m) worst case
 * Uses bad character and good suffix heuristics to skip text
 */
export function boyerMooreSearch(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  if (m === 0) return results;

  // Build bad character table
  const badChar = buildBadCharTable(pattern);

  // Build good suffix table
  const goodSuffix = buildGoodSuffixTable(pattern);

  let i = 0; // alignment of pattern with text

  while (i <= n - m) {
    let j = m - 1; // start from end of pattern

    while (j >= 0 && pattern[j] === text[i + j]) {
      j--;
    }

    if (j < 0) {
      // Found a match
      results.push(i);
      i += goodSuffix[0];
    } else {
      // Mismatch - shift by max of bad char and good suffix
      const charCode = text.charCodeAt(i + j);
      const badCharShift = Math.max(1, j - (badChar[charCode] ?? -1));
      const goodSuffixShift = goodSuffix[j + 1] || 1;
      i += Math.max(badCharShift, goodSuffixShift);
    }
  }

  return results;
}

function buildBadCharTable(pattern: string): number[] {
  const table: number[] = new Array(256).fill(-1);
  const m = pattern.length;

  for (let i = 0; i < m; i++) {
    table[pattern.charCodeAt(i)] = i;
  }

  return table;
}

function buildGoodSuffixTable(pattern: string): number[] {
  const m = pattern.length;
  const suffix = computeSuffixes(pattern);
  const table = new Array(m + 1).fill(m);

  // Case 1: matching suffix exists elsewhere in pattern
  for (let i = m - 1; i >= 0; i--) {
    if (suffix[i] === i + 1) {
      for (let j = 0; j < m - 1 - i; j++) {
        if (table[j] === m) {
          table[j] = m - 1 - i;
        }
      }
    }
  }

  // Case 2: part of matching suffix occurs at beginning
  for (let i = 0; i < m - 1; i++) {
    table[m - 1 - suffix[i]] = m - 1 - i;
  }

  return table;
}

function computeSuffixes(pattern: string): number[] {
  const m = pattern.length;
  const suffix = new Array(m).fill(0);
  suffix[m - 1] = m;

  let g = m - 1;
  let f = 0;

  for (let i = m - 2; i >= 0; i--) {
    if (i > g && suffix[i + m - 1 - f] < i - g) {
      suffix[i] = suffix[i + m - 1 - f];
    } else {
      if (i < g) {
        g = i;
      }
      f = i;
      while (g >= 0 && pattern[g] === pattern[g + m - 1 - f]) {
        g--;
      }
      suffix[i] = f - g;
    }
  }

  return suffix;
}

/**
 * Rabin-Karp - O(n+m) average, O(n*m) worst
 * Uses rolling hash for fast substring comparison
 */
export function rabinKarpSearch(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  if (m === 0 || m > n) return results;

  const prime = 101;
  const base = 256;

  // Calculate hash of pattern and first window of text
  let patternHash = 0;
  let textHash = 0;
  let h = 1;

  // h = base^(m-1) % prime
  for (let i = 0; i < m - 1; i++) {
    h = (h * base) % prime;
  }

  // Calculate initial hashes
  for (let i = 0; i < m; i++) {
    patternHash = (base * patternHash + pattern.charCodeAt(i)) % prime;
    textHash = (base * textHash + text.charCodeAt(i)) % prime;
  }

  // Slide pattern over text
  for (let i = 0; i <= n - m; i++) {
    if (patternHash === textHash) {
      // Check character by character
      let match = true;
      for (let j = 0; j < m; j++) {
        if (text[i + j] !== pattern[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        results.push(i);
      }
    }

    // Calculate hash for next window
    if (i < n - m) {
      textHash =
        (base * (textHash - text.charCodeAt(i) * h) + text.charCodeAt(i + m)) % prime;
      if (textHash < 0) {
        textHash += prime;
      }
    }
  }

  return results;
}

/**
 * Horspool - Simplified Boyer-Moore
 * Uses only bad character rule, simpler but still fast
 */
export function horspoolSearch(text: string, pattern: string): number[] {
  const results: number[] = [];
  const n = text.length;
  const m = pattern.length;

  if (m === 0 || m > n) return results;

  // Build shift table
  const shift: number[] = new Array(256).fill(m);
  for (let i = 0; i < m - 1; i++) {
    shift[pattern.charCodeAt(i)] = m - 1 - i;
  }

  let i = 0;
  while (i <= n - m) {
    let j = m - 1;
    while (j >= 0 && text[i + j] === pattern[j]) {
      j--;
    }

    if (j < 0) {
      results.push(i);
      i += shift[text.charCodeAt(i + m - 1)] || m;
    } else {
      i += shift[text.charCodeAt(i + m - 1)] || 1;
    }
  }

  return results;
}

// Export all algorithms for benchmarking
export const algorithms = {
  naive: naiveSearch,
  kmp: kmpSearch,
  boyerMoore: boyerMooreSearch,
  rabinKarp: rabinKarpSearch,
  horspool: horspoolSearch,
};

// Default export for evolution seed
export default boyerMooreSearch;
