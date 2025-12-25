use crate::StringSearch;
use std::ptr;

pub struct EvolvedSearch {
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
    }

    #[inline(always)]
    fn search_length_1(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let target = pattern[0];
        let text_ptr = text.as_ptr();
        let text_len = text.len();

        results.reserve(text_len / 16);

        let mut i = 0;
        while i < text_len {
            unsafe {
                if *text_ptr.add(i) == target {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_length_2(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 2 { return results; }

        let p0 = pattern[0];
        let p1 = pattern[1];
        let pair = u16::from_ne_bytes([p0, p1]);

        results.reserve((n - 1) / 32);

        let text_ptr = text.as_ptr();
        let mut i = 0;

        while i <= n - 2 {
            unsafe {
                let text_pair = u16::from_ne_bytes([
                    *text_ptr.add(i),
                    *text_ptr.add(i + 1)
                ]);
                if text_pair == pair {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_length_3(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 3 { return results; }

        let p0 = pattern[0];
        let p1 = pattern[1];
        let p2 = pattern[2];

        results.reserve((n - 2) / 32);

        let text_ptr = text.as_ptr();
        let mut i = 0;

        while i <= n - 3 {
            unsafe {
                if *text_ptr.add(i) == p0
                    && *text_ptr.add(i + 1) == p1
                    && *text_ptr.add(i + 2) == p2 {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_length_4(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 4 { return results; }

        let pattern_u32 = u32::from_ne_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);

        results.reserve((n - 3) / 32);

        let text_ptr = text.as_ptr();
        let mut i = 0;

        while i <= n - 4 {
            unsafe {
                let text_u32 = u32::from_ne_bytes([
                    *text_ptr.add(i),
                    *text_ptr.add(i + 1),
                    *text_ptr.add(i + 2),
                    *text_ptr.add(i + 3)
                ]);
                if text_u32 == pattern_u32 {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_length_5_to_8(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        let m = pattern.len();
        if n < m { return results; }

        let pattern_u64 = unsafe {
            let mut bytes = [0u8; 8];
            ptr::copy_nonoverlapping(pattern.as_ptr(), bytes.as_mut_ptr(), m);
            u64::from_ne_bytes(bytes)
        };

        results.reserve((n - m + 1) / 32);

        let text_ptr = text.as_ptr();
        let mut i = 0;
        let end = n - m + 1;

        while i < end {
            unsafe {
                let mut text_bytes = [0u8; 8];
                ptr::copy_nonoverlapping(text_ptr.add(i), text_bytes.as_mut_ptr(), m);
                let text_u64 = u64::from_ne_bytes(text_bytes);

                if text_u64 == pattern_u64 {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_long_pattern(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        let mut shift = [m; 256];
        let pattern_ptr = pattern.as_ptr();

        for i in 0..(m - 1) {
            unsafe {
                let byte = *pattern_ptr.add(i) as usize;
                shift[byte] = m - 1 - i;
            }
        }

        let mut results = Vec::new();
        results.reserve((n / m).max(1));

        let text_ptr = text.as_ptr();
        let last_pattern_byte = unsafe { *pattern_ptr.add(m - 1) };
        let first_pattern_byte = unsafe { *pattern_ptr };
        let second_to_last_pattern = unsafe { *pattern_ptr.add(m - 2) };
        let end = n - m;
        let m_minus_1 = m - 1;

        let mut i = 0;
        while i <= end {
            unsafe {
                let last_text_byte = *text_ptr.add(i + m_minus_1);

                if last_text_byte == last_pattern_byte {
                    let first_text_byte = *text_ptr.add(i);
                    if first_text_byte == first_pattern_byte {
                        let second_to_last_text = *text_ptr.add(i + m - 2);

                        if second_to_last_text == second_to_last_pattern {
                            let mut matched = true;
                            let mut j = 1;
                            while j < m_minus_1 {
                                if j + 1 < m_minus_1 {
                                    if *text_ptr.add(i + j) != *pattern_ptr.add(j)
                                        || *text_ptr.add(i + j + 1) != *pattern_ptr.add(j + 1) {
                                        matched = false;
                                        break;
                                    }
                                    j += 2;
                                } else {
                                    if *text_ptr.add(i + j) != *pattern_ptr.add(j) {
                                        matched = false;
                                    }
                                    j += 1;
                                }
                            }

                            if matched {
                                results.push(i);
                            }
                        }
                    }
                }
                i += shift[last_text_byte as usize].max(1);
            }
        }
        results
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

        if m == 0 { return Vec::new(); }
        if m > n { return Vec::new(); }

        match m {
            1 => self.search_length_1(text, pattern),
            2 => self.search_length_2(text, pattern),
            3 => self.search_length_3(text, pattern),
            4 => self.search_length_4(text, pattern),
            5..=8 => self.search_length_5_to_8(text, pattern),
            _ => self.search_long_pattern(text, pattern),
        }
    }
}
