use crate::StringSearch;

pub struct EvolvedSearch {
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
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

        if m == 0 { return Vec::new(); }
        if m > n { return Vec::new(); }
        if m == 1 {
            let target = pattern[0];
            return text.iter().enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        if m == 2 {
            let first = pattern[0];
            let second = pattern[1];
            let mut results = Vec::new();
            for i in 0..=n - 2 {
                if text[i] == first && text[i + 1] == second {
                    results.push(i);
                }
            }
            return results;
        }

        let mut shift = [m; 256];
        for (i, &byte) in pattern.iter().enumerate().take(m - 1) {
            shift[byte as usize] = m - 1 - i;
        }

        const PRIME: u64 = 16777619;
        let mut pattern_hash: u64 = 0;
        for &byte in pattern.iter() {
            pattern_hash = pattern_hash.wrapping_mul(PRIME).wrapping_add(byte as u64);
        }

        let mut results = Vec::new();
        let last_pattern_byte = pattern[m - 1];
        let first_pattern_byte = pattern[0];
        let second_pattern_byte = pattern[1];

        let mut i = 0;
        while i <= n - m {
            let last_text_byte = text[i + m - 1];

            if last_text_byte == last_pattern_byte {
                if text[i] == first_pattern_byte && text[i + 1] == second_pattern_byte {
                    let mut text_hash: u64 = 0;
                    for j in 0..m {
                        text_hash = text_hash.wrapping_mul(PRIME).wrapping_add(text[i + j] as u64);
                    }

                    if text_hash == pattern_hash {
                        if &text[i..i + m] == pattern {
                            results.push(i);
                        }
                    }
                }
                i += 1;
            } else {
                i += shift[last_text_byte as usize];
            }
        }

        results
    }
}
