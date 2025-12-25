use crate::StringSearch;

pub struct EvolvedSearch {
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
    }

    #[inline(always)]
    fn search_length_2(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
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
    fn search_length_3(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 3 { return results; }

        let p0 = pattern[0];
        let p1 = pattern[1];
        let p2 = pattern[2];

        let mut i = 0;
        while i <= n - 3 {
            unsafe {
                if *text.get_unchecked(i) == p0
                    && *text.get_unchecked(i + 1) == p1
                    && *text.get_unchecked(i + 2) == p2 {
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

        let mut i = 0;
        while i <= n - 4 {
            unsafe {
                let text_u32 = u32::from_ne_bytes([
                    *text.get_unchecked(i),
                    *text.get_unchecked(i + 1),
                    *text.get_unchecked(i + 2),
                    *text.get_unchecked(i + 3)
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
    fn search_long_pattern(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        let mut shift = [m; 256];
        for i in 0..(m - 1) {
            shift[pattern[i] as usize] = m - 1 - i;
        }

        let mut results = Vec::new();
        let mut i = 0;
        let last_pattern_byte = pattern[m - 1];
        let first_pattern_byte = pattern[0];

        while i <= n - m {
            unsafe {
                let last_text_byte = *text.get_unchecked(i + m - 1);
                if last_text_byte == last_pattern_byte {
                    if *text.get_unchecked(i) == first_pattern_byte {
                        let mut matched = true;
                        for j in 1..(m - 1) {
                            if *text.get_unchecked(i + j) != *pattern.get_unchecked(j) {
                                matched = false;
                                break;
                            }
                        }
                        if matched {
                            results.push(i);
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

        match m {
            2 => self.search_length_2(text, pattern),
            3 => self.search_length_3(text, pattern),
            4 => self.search_length_4(text, pattern),
            _ => self.search_long_pattern(text, pattern),
        }
    }
}
