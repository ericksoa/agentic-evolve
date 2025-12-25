use crate::StringSearch;

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

        let text_ptr = text.as_ptr();
        let mut i = 0;
        while i <= n - 2 {
            unsafe {
                if *text_ptr.add(i) == p0 && *text_ptr.add(i + 1) == p1 {
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
    fn search_length_4(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
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
    fn search_length_8(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        if n < 8 { return results; }

        let pattern_u64 = u64::from_ne_bytes([
            pattern[0], pattern[1], pattern[2], pattern[3],
            pattern[4], pattern[5], pattern[6], pattern[7],
        ]);

        let text_ptr = text.as_ptr();
        let mut i = 0;
        while i <= n - 8 {
            unsafe {
                let text_u64 = u64::from_ne_bytes([
                    *text_ptr.add(i),
                    *text_ptr.add(i + 1),
                    *text_ptr.add(i + 2),
                    *text_ptr.add(i + 3),
                    *text_ptr.add(i + 4),
                    *text_ptr.add(i + 5),
                    *text_ptr.add(i + 6),
                    *text_ptr.add(i + 7),
                ]);
                if text_u64 == pattern_u64 {
                    results.push(i);
                }
            }
            i += 1;
        }
        results
    }

    #[inline(never)]
    fn search_general(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        let m = pattern.len();

        let mut shift = [m; 256];
        for i in 0..(m - 1) {
            shift[pattern[i] as usize] = m - 1 - i;
        }

        let text_ptr = text.as_ptr();
        let pattern_ptr = pattern.as_ptr();
        let last_byte = pattern[m - 1];
        let first_byte = pattern[0];

        let mut i = 0;
        while i <= n - m {
            unsafe {
                let text_last = *text_ptr.add(i + m - 1);
                if text_last == last_byte && *text_ptr.add(i) == first_byte {
                    let mut matched = true;
                    for j in 1..(m - 1) {
                        if *text_ptr.add(i + j) != *pattern_ptr.add(j) {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        results.push(i);
                    }
                }
                i += shift[text_last as usize].max(1);
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

        if m == 0 || m > n {
            return Vec::new();
        }

        match m {
            1 => self.search_length_1(text, pattern),
            2 => self.search_length_2(text, pattern),
            3 => self.search_length_3(text, pattern),
            4 => self.search_length_4(text, pattern),
            8 => self.search_length_8(text, pattern),
            _ => self.search_general(text, pattern),
        }
    }
}
