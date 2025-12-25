use crate::StringSearch;

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
    #[inline(always)]
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
            let mut results = Vec::new();
            let text_ptr = text.as_ptr();
            unsafe {
                for i in 0..n {
                    if *text_ptr.add(i) == target {
                        results.push(i);
                    }
                }
            }
            return results;
        }

        if m == 2 {
            return Self::search_two_byte(text, pattern);
        }

        Self::search_multi_byte(text, pattern)
    }
}

impl EvolvedSearch {
    #[inline]
    fn search_two_byte(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let mut results = Vec::new();
        let first = pattern[0];
        let second = pattern[1];

        unsafe {
            let text_ptr = text.as_ptr();
            let end = n - 1;

            for i in 0..end {
                if *text_ptr.add(i) == first && *text_ptr.add(i + 1) == second {
                    results.push(i);
                }
            }
        }
        results
    }

    #[inline]
    fn search_multi_byte(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        let mut shift = [m; 256];
        let pattern_ptr = pattern.as_ptr();
        let text_ptr = text.as_ptr();

        unsafe {
            for i in 0..(m - 1) {
                let byte_val = *pattern_ptr.add(i) as usize;
                shift[byte_val] = m - 1 - i;
            }
        }

        let last_pattern_byte: u8;
        let first_pattern_byte: u8;
        let second_pattern_byte: u8;

        unsafe {
            last_pattern_byte = *pattern_ptr.add(m - 1);
            first_pattern_byte = *pattern_ptr;
            second_pattern_byte = if m > 2 { *pattern_ptr.add(1) } else { 0 };
        }

        let mut results = Vec::new();
        let mut i = 0;
        let end = n - m;

        unsafe {
            while i <= end {
                let text_last = *text_ptr.add(i + m - 1);

                if text_last == last_pattern_byte {
                    let text_first = *text_ptr.add(i);
                    if text_first == first_pattern_byte {
                        let mut matched = true;
                        if m > 2 {
                            let text_second = *text_ptr.add(i + 1);
                            if text_second != second_pattern_byte {
                                matched = false;
                            }
                        }

                        if matched {
                            matched = Self::compare_bytes_fast(
                                text_ptr.add(i),
                                pattern_ptr,
                                m,
                            );
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

    #[inline]
    unsafe fn compare_bytes_fast(
        text_ptr: *const u8,
        pattern_ptr: *const u8,
        len: usize,
    ) -> bool {
        let mut j = 0;

        let unroll_end = len.saturating_sub(4);
        while j < unroll_end {
            if *text_ptr.add(j) != *pattern_ptr.add(j)
                || *text_ptr.add(j + 1) != *pattern_ptr.add(j + 1)
                || *text_ptr.add(j + 2) != *pattern_ptr.add(j + 2)
                || *text_ptr.add(j + 3) != *pattern_ptr.add(j + 3)
            {
                return false;
            }
            j += 4;
        }

        while j < len {
            if *text_ptr.add(j) != *pattern_ptr.add(j) {
                return false;
            }
            j += 1;
        }

        true
    }
}
