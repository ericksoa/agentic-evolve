use crate::StringSearch;

pub struct EvolvedSearch {}

impl EvolvedSearch {
    pub fn new() -> Self { Self {} }

    #[inline(always)]
    fn build_bloom_filter(pattern: &[u8]) -> u64 {
        let mut bloom = 0u64;
        for &byte in pattern {
            bloom |= 1u64 << (byte as u64 & 0x3F);
        }
        bloom
    }

    #[inline(always)]
    fn bloom_check(bloom: u64, byte: u8) -> bool {
        bloom & (1u64 << (byte as u64 & 0x3F)) != 0
    }

    #[inline(always)]
    fn search_2(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 2 { return results; }

        let p0 = pattern[0];
        let p1 = pattern[1];
        let pair = u16::from_ne_bytes([p0, p1]);

        let mut i = 0;
        while i <= n - 2 {
            unsafe {
                let text_pair = u16::from_ne_bytes([
                    *text.get_unchecked(i),
                    *text.get_unchecked(i + 1)
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
    fn search_3(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 3 { return results; }

        let p0 = pattern[0];
        let p1 = pattern[1];
        let p2 = pattern[2];

        let text_ptr = text.as_ptr();
        let mut i = 0;
        while i <= n - 3 {
            unsafe {
                if *text_ptr.add(i) == p0
                    && *text_ptr.add(i + 1) == p1
                    && *text_ptr.add(i + 2) == p2
                {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(always)]
    fn search_4(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 4 { return results; }

        let pattern_u32 = u32::from_ne_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);

        let text_ptr = text.as_ptr();
        let mut i = 0;
        while i <= n - 4 {
            unsafe {
                let text_u32 = u32::from_ne_bytes([
                    *text_ptr.add(i),
                    *text_ptr.add(i + 1),
                    *text_ptr.add(i + 2),
                    *text_ptr.add(i + 3),
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
    fn build_shift_table(pattern: &[u8]) -> [usize; 256] {
        let m = pattern.len();
        let mut shift = [m; 256];
        for i in 0..(m - 1) {
            shift[pattern[i] as usize] = m - 1 - i;
        }
        shift
    }

    #[inline(always)]
    fn build_secondary_shift_table(pattern: &[u8]) -> [usize; 256] {
        let m = pattern.len();
        let mut shift = [m; 256];
        if m < 2 {
            return shift;
        }

        for i in 0..(m - 2) {
            shift[pattern[i] as usize] = m - 2 - i;
        }
        shift
    }

    #[inline(always)]
    fn unrolled_match(text: &[u8], pattern: &[u8], pos: usize, m: usize) -> bool {
        unsafe {
            let text_ptr = text.as_ptr().add(pos);
            let pattern_ptr = pattern.as_ptr();

            let mut j = 1;
            let end = m - 1;

            while j + 7 < end {
                if *text_ptr.add(j) != *pattern_ptr.add(j)
                    || *text_ptr.add(j + 1) != *pattern_ptr.add(j + 1)
                    || *text_ptr.add(j + 2) != *pattern_ptr.add(j + 2)
                    || *text_ptr.add(j + 3) != *pattern_ptr.add(j + 3)
                    || *text_ptr.add(j + 4) != *pattern_ptr.add(j + 4)
                    || *text_ptr.add(j + 5) != *pattern_ptr.add(j + 5)
                    || *text_ptr.add(j + 6) != *pattern_ptr.add(j + 6)
                    || *text_ptr.add(j + 7) != *pattern_ptr.add(j + 7)
                {
                    return false;
                }
                j += 8;
            }

            while j < end {
                if *text_ptr.add(j) != *pattern_ptr.add(j) {
                    return false;
                }
                j += 1;
            }
            true
        }
    }

    #[inline]
    fn search_general(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        let shift_table = Self::build_shift_table(pattern);
        let secondary_shift = Self::build_secondary_shift_table(pattern);
        let bloom = Self::build_bloom_filter(pattern);

        let mut results = Vec::with_capacity((n / m).min(256));

        unsafe {
            let text_ptr = text.as_ptr();
            let pattern_ptr = pattern.as_ptr();

            let last_byte = *pattern_ptr.add(m - 1);
            let first_byte = *pattern_ptr;
            let second_last_byte = if m >= 2 { *pattern_ptr.add(m - 2) } else { 0 };

            let mut i = 0;
            while i <= n - m {
                let text_last = *text_ptr.add(i + m - 1);

                if !Self::bloom_check(bloom, text_last) {
                    i += shift_table[text_last as usize].max(1);
                    continue;
                }

                if text_last == last_byte {
                    let text_first = *text_ptr.add(i);

                    if text_first == first_byte {
                        if m >= 2 {
                            let text_second_last = *text_ptr.add(i + m - 2);
                            if text_second_last != second_last_byte {
                                i += secondary_shift[text_second_last as usize].max(1);
                                continue;
                            }
                        }

                        if Self::unrolled_match(text, pattern, i, m) {
                            results.push(i);
                            i += 1;
                        } else {
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                } else {
                    i += shift_table[text_last as usize].max(1);
                }
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
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 || m > n {
            return Vec::new();
        }

        if m == 1 {
            let target = pattern[0];
            return text
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| if b == target { Some(i) } else { None })
                .collect();
        }

        match m {
            2 => Self::search_2(text, pattern),
            3 => Self::search_3(text, pattern),
            4 => Self::search_4(text, pattern),
            _ => Self::search_general(text, pattern),
        }
    }
}
