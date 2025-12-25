use crate::StringSearch;
use std::ptr;

pub struct EvolvedSearch {}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for EvolvedSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for EvolvedSearch {
    #[inline]
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        match m {
            1 => Self::search_u8(text, pattern),
            2 => Self::search_u16(text, pattern),
            3 => Self::search_u24(text, pattern),
            4 => Self::search_u32(text, pattern),
            5..=8 => Self::search_u64_masked(text, pattern),
            _ => Self::search_horspool_u64(text, pattern),
        }
    }
}

impl EvolvedSearch {
    #[inline]
    fn search_u8(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let target = pattern[0];
        for (i, &byte) in text.iter().enumerate() {
            if byte == target {
                results.push(i);
            }
        }
        results
    }

    #[inline]
    fn search_u16(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let pattern_word = u16::from_le_bytes([pattern[0], pattern[1]]);

        unsafe {
            let text_ptr = text.as_ptr();
            let n = text.len();
            let end = n.saturating_sub(1);

            for i in 0..end {
                let word = ptr::read_unaligned(text_ptr.add(i) as *const u16);
                if word == pattern_word {
                    results.push(i);
                }
            }
        }
        results
    }

    #[inline]
    fn search_u24(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let p0 = pattern[0];
        let p1 = pattern[1];
        let p2 = pattern[2];

        unsafe {
            let text_ptr = text.as_ptr();
            let n = text.len();
            let end = n.saturating_sub(2);

            for i in 0..end {
                if *text_ptr.add(i) == p0
                    && *text_ptr.add(i + 1) == p1
                    && *text_ptr.add(i + 2) == p2
                {
                    results.push(i);
                }
            }
        }
        results
    }

    #[inline]
    fn search_u32(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let pattern_word = u32::from_le_bytes([
            pattern[0],
            pattern[1],
            pattern[2],
            pattern[3],
        ]);

        unsafe {
            let text_ptr = text.as_ptr();
            let n = text.len();
            let end = n.saturating_sub(3);

            for i in 0..end {
                let word = ptr::read_unaligned(text_ptr.add(i) as *const u32);
                if word == pattern_word {
                    results.push(i);
                }
            }
        }
        results
    }

    #[inline]
    fn search_u64_masked(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let m = pattern.len();

        let mut pattern_word = 0u64;
        let mut mask = 0u64;
        for i in 0..m {
            pattern_word |= (pattern[i] as u64) << (i * 8);
            mask |= 0xFFu64 << (i * 8);
        }

        unsafe {
            let text_ptr = text.as_ptr();
            let n = text.len();
            let end = n.saturating_sub(m - 1);

            for i in 0..end {
                let word = ptr::read_unaligned(text_ptr.add(i) as *const u64) & mask;
                if word == pattern_word {
                    results.push(i);
                }
            }
        }
        results
    }

    #[inline]
    fn search_horspool_u64(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        let mut results = Vec::new();
        let mut shift = [m; 256];

        unsafe {
            let pattern_ptr = pattern.as_ptr();
            for i in 0..(m - 1) {
                let byte_val = *pattern_ptr.add(i) as usize;
                shift[byte_val] = m - 1 - i;
            }
        }

        let first_u64_len = m.min(8);
        let mut pattern_prefix = 0u64;
        let mut prefix_mask = 0u64;

        unsafe {
            let pattern_ptr = pattern.as_ptr();
            for i in 0..first_u64_len {
                pattern_prefix |= (*pattern_ptr.add(i) as u64) << (i * 8);
                prefix_mask |= 0xFFu64 << (i * 8);
            }
        }

        unsafe {
            let text_ptr = text.as_ptr();
            let pattern_ptr = pattern.as_ptr();
            let last_byte = *pattern_ptr.add(m - 1);
            let mut i = 0;
            let end = n - m;

            while i <= end {
                let text_last = *text_ptr.add(i + m - 1);

                if text_last == last_byte {
                    let text_prefix = ptr::read_unaligned(text_ptr.add(i) as *const u64) & prefix_mask;

                    if text_prefix == pattern_prefix {
                        let mut matched = true;
                        let mut j = 8;
                        while j < m && matched {
                            if *text_ptr.add(i + j) != *pattern_ptr.add(j) {
                                matched = false;
                            }
                            j += 1;
                        }

                        if matched {
                            results.push(i);
                        }
                    }
                }

                let shift_val = shift[text_last as usize];
                i = i.saturating_add(shift_val).max(i + 1);
            }
        }

        results
    }
}
