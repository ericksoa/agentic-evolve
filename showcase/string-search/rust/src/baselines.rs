//! Baseline String Search Algorithms
//!
//! These are well-known, battle-tested string search algorithms.
//! Our evolved algorithm will be benchmarked against these.

use crate::StringSearch;

/// Naive O(n*m) string search
pub struct NaiveSearch;

impl NaiveSearch {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NaiveSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for NaiveSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        let mut results = Vec::new();
        for i in 0..=n - m {
            let mut matched = true;
            for j in 0..m {
                if text[i + j] != pattern[j] {
                    matched = false;
                    break;
                }
            }
            if matched {
                results.push(i);
            }
        }
        results
    }
}

/// Knuth-Morris-Pratt O(n+m) string search
pub struct KMPSearch;

impl KMPSearch {
    pub fn new() -> Self {
        Self
    }

    fn build_failure_table(pattern: &[u8]) -> Vec<usize> {
        let m = pattern.len();
        let mut failure = vec![0; m];

        let mut j = 0;
        for i in 1..m {
            while j > 0 && pattern[i] != pattern[j] {
                j = failure[j - 1];
            }
            if pattern[i] == pattern[j] {
                j += 1;
            }
            failure[i] = j;
        }

        failure
    }
}

impl Default for KMPSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for KMPSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        let failure = Self::build_failure_table(pattern);
        let mut results = Vec::new();
        let mut j = 0;

        for (i, &byte) in text.iter().enumerate() {
            while j > 0 && byte != pattern[j] {
                j = failure[j - 1];
            }
            if byte == pattern[j] {
                j += 1;
            }
            if j == m {
                results.push(i + 1 - m);
                j = failure[j - 1];
            }
        }

        results
    }
}

/// Boyer-Moore string search with bad character heuristic
/// Using simplified version (bad character only) for reliability
pub struct BoyerMooreSearch;

impl BoyerMooreSearch {
    pub fn new() -> Self {
        Self
    }

    fn build_bad_char_table(pattern: &[u8]) -> [isize; 256] {
        let mut table = [-1isize; 256];

        for (i, &byte) in pattern.iter().enumerate() {
            table[byte as usize] = i as isize;
        }

        table
    }
}

impl Default for BoyerMooreSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for BoyerMooreSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len() as isize;
        let m = pattern.len() as isize;

        if m == 0 || m > n {
            return Vec::new();
        }

        let bad_char = Self::build_bad_char_table(pattern);
        let mut results = Vec::new();
        let mut s: isize = 0;

        while s <= n - m {
            let mut j = m - 1;

            // Keep reducing j while characters match
            while j >= 0 && pattern[j as usize] == text[(s + j) as usize] {
                j -= 1;
            }

            if j < 0 {
                // Pattern found
                results.push(s as usize);

                // Shift pattern to align next character in text with last occurrence in pattern
                if s + m < n {
                    s += m - bad_char[text[(s + m) as usize] as usize];
                } else {
                    s += 1;
                }
            } else {
                // Shift pattern so bad character aligns with last occurrence in pattern
                let shift = j - bad_char[text[(s + j) as usize] as usize];
                s += shift.max(1);
            }
        }

        results
    }
}

/// Horspool (simplified Boyer-Moore) string search
pub struct HorspoolSearch;

impl HorspoolSearch {
    pub fn new() -> Self {
        Self
    }

    fn build_shift_table(pattern: &[u8]) -> [usize; 256] {
        let m = pattern.len();
        let mut table = [m; 256];

        for (i, &byte) in pattern.iter().enumerate().take(m - 1) {
            table[byte as usize] = m - 1 - i;
        }

        table
    }
}

impl Default for HorspoolSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for HorspoolSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        let shift = Self::build_shift_table(pattern);
        let mut results = Vec::new();
        let mut i = 0;

        while i <= n - m {
            let mut j = m - 1;

            while pattern[j] == text[i + j] {
                if j == 0 {
                    results.push(i);
                    break;
                }
                j -= 1;
            }

            i += shift[text[i + m - 1] as usize].max(1);
        }

        results
    }
}

/// Two-Way string search (used by glibc memmem)
pub struct TwoWaySearch;

impl TwoWaySearch {
    pub fn new() -> Self {
        Self
    }

    fn critical_factorization(pattern: &[u8]) -> (usize, usize) {
        let m = pattern.len();

        // Find the maximal suffix using lexicographic order
        let (mut i, mut j, mut k, mut p) = (0usize, 1usize, 1usize, 1usize);

        while j + k <= m {
            let a = pattern[i + k - 1];
            let b = pattern[j + k - 1];

            if a < b {
                j += k;
                k = 1;
                p = j - i;
            } else if a == b {
                if k == p {
                    j += p;
                    k = 1;
                } else {
                    k += 1;
                }
            } else {
                i = j;
                j = i + 1;
                k = 1;
                p = 1;
            }
        }

        let ell = i;
        let period = p;

        // Check if pattern[:ell] is a suffix of pattern[ell:ell+period]
        let memory = if ell < period && &pattern[..ell] == &pattern[period..period + ell] {
            period
        } else {
            0
        };

        (ell, if memory > 0 { period } else { ell + 1 }.max(m - ell))
    }
}

impl Default for TwoWaySearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for TwoWaySearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        let (ell, period) = Self::critical_factorization(pattern);
        let mut results = Vec::new();
        let mut i = 0;

        while i <= n - m {
            // Match right part
            let mut j = ell;
            while j < m && pattern[j] == text[i + j] {
                j += 1;
            }

            if j < m {
                i += j - ell + 1;
                continue;
            }

            // Match left part
            j = ell;
            while j > 0 && pattern[j - 1] == text[i + j - 1] {
                j -= 1;
            }

            if j == 0 {
                results.push(i);
            }

            i += period;
        }

        results
    }
}
