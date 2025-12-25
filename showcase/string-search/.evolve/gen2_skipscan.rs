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
            return Self::search_single_byte(text, pattern);
        }

        if m == 2 {
            return Self::search_two_bytes(text, pattern);
        }

        Self::search_multi_byte(text, pattern)
    }
}

impl EvolvedSearch {
    #[inline]
    fn search_single_byte(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let target = pattern[0];
        let mut results = Vec::new();
        let text_ptr = text.as_ptr();
        let n = text.len();

        unsafe {
            for i in 0..n {
                if *text_ptr.add(i) == target {
                    results.push(i);
                }
            }
        }

        results
    }

    #[inline]
    fn search_two_bytes(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let n = text.len();
        let p0 = pattern[0];
        let p1 = pattern[1];

        unsafe {
            let text_ptr = text.as_ptr();
            let end = n - 1;
            let mut i = 0;

            while i < end {
                if *text_ptr.add(i) == p0 && *text_ptr.add(i + 1) == p1 {
                    results.push(i);
                }
                i += 1;
            }
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

    #[inline(never)]
    fn search_multi_byte(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();
        let mut results = Vec::new();

        let shift = Self::build_shift_table(pattern);

        unsafe {
            let text_ptr = text.as_ptr();
            let pattern_ptr = pattern.as_ptr();

            let first_byte = *pattern_ptr;
            let last_byte = *pattern_ptr.add(m - 1);
            let second_last_byte = *pattern_ptr.add(m - 2);

            let end = n.saturating_sub(m);
            let mut i = 0;

            while i <= end {
                let last_text = *text_ptr.add(i + m - 1);

                if last_text == last_byte {
                    let second_last_text = *text_ptr.add(i + m - 2);

                    if second_last_text == second_last_byte {
                        let first_text = *text_ptr.add(i);

                        if first_text == first_byte {
                            if Self::compare_bytes_fast(
                                text_ptr.add(i),
                                pattern_ptr,
                                m,
                            ) {
                                results.push(i);
                            }
                            i += 1;
                        } else {
                            i += shift[last_text as usize];
                        }
                    } else {
                        i += shift[second_last_text as usize];
                    }
                } else {
                    i += shift[last_text as usize].max(1);
                }
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

        let unroll_end = len.saturating_sub(8);
        while j < unroll_end {
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

        while j < len {
            if *text_ptr.add(j) != *pattern_ptr.add(j) {
                return false;
            }
            j += 1;
        }

        true
    }
}
