use crate::StringSearch;

pub struct EvolvedSearch {
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
    }

    #[inline(always)]
    fn memchr_from(needle: u8, haystack: &[u8], start: usize) -> Option<usize> {
        haystack[start..].iter().position(|&b| b == needle).map(|pos| pos + start)
    }

    #[inline(always)]
    fn verify_match(text: &[u8], pattern: &[u8], pos: usize) -> bool {
        let m = pattern.len();
        if pos + m > text.len() {
            return false;
        }

        if m >= 16 {
            if text.len() >= pos + 8 && pattern.len() >= 8 {
                let text_u64 = u64::from_ne_bytes(text[pos..pos + 8].try_into().unwrap());
                let pattern_u64 = u64::from_ne_bytes(pattern[0..8].try_into().unwrap());
                if text_u64 != pattern_u64 {
                    return false;
                }
            }
            if text.len() >= pos + m && pattern.len() >= m && m >= 16 {
                let text_u64_end = u64::from_ne_bytes(text[pos + m - 8..pos + m].try_into().unwrap());
                let pattern_u64_end = u64::from_ne_bytes(pattern[m - 8..m].try_into().unwrap());
                if text_u64_end != pattern_u64_end {
                    return false;
                }
            }
        }

        &text[pos..pos + m] == pattern
    }
}

impl Default for EvolvedSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for EvolvedSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Vec::new();
        }
        if m > n {
            return Vec::new();
        }

        if m == 1 {
            let target = pattern[0];
            return text.iter().enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        if m == 2 {
            let mut results = Vec::new();
            let target = u16::from_ne_bytes([pattern[0], pattern[1]]);
            let mut i = 0;
            while i <= n - 2 {
                let chunk = u16::from_ne_bytes([text[i], text[i + 1]]);
                if chunk == target {
                    results.push(i);
                }
                i += 1;
            }
            return results;
        }

        let mut results = Vec::new();
        let first_byte = pattern[0];
        let last_byte = pattern[m - 1];

        if m >= 8 {
            let mut i = 0;
            while i <= n - m {
                if let Some(pos) = Self::memchr_from(first_byte, text, i) {
                    if pos > n - m {
                        break;
                    }
                    i = pos;

                    if text[i + m - 1] == last_byte {
                        if Self::verify_match(text, pattern, i) {
                            results.push(i);
                        }
                    }
                    i += 1;
                } else {
                    break;
                }
            }
        } else {
            let mut i = 0;
            while i <= n - m {
                if let Some(pos) = Self::memchr_from(first_byte, text, i) {
                    if pos > n - m {
                        break;
                    }
                    i = pos;

                    if &text[i..i + m] == pattern {
                        results.push(i);
                    }
                    i += 1;
                } else {
                    break;
                }
            }
        }

        results
    }
}
