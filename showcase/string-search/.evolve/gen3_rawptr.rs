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

        if m == 0 || m > n {
            return Vec::new();
        }

        match m {
            1 => Self::search_len1(text, pattern),
            2 => Self::search_len2(text, pattern),
            3 => Self::search_len3(text, pattern),
            4 => Self::search_len4(text, pattern),
            _ => Self::search_generic(text, pattern),
        }
    }
}

impl EvolvedSearch {
    #[inline]
    fn search_len1(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let target = unsafe { *pattern.as_ptr() };
        let text_ptr = text.as_ptr();
        let text_len = text.len();

        unsafe {
            let mut i = 0;
            let mut ptr = text_ptr;
            let end_ptr = text_ptr.add(text_len);

            while ptr < end_ptr {
                if *ptr == target {
                    results.push(i);
                }
                ptr = ptr.add(1);
                i += 1;
            }
        }

        results
    }

    #[inline]
    fn search_len2(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let text_len = text.len();

        if text_len < 2 {
            return results;
        }

        let p0 = unsafe { *pattern.as_ptr() };
        let p1 = unsafe { *pattern.as_ptr().add(1) };
        let text_ptr = text.as_ptr();

        unsafe {
            let mut i = 0;
            let limit = text_len - 1;

            while i < limit {
                let ptr = text_ptr.add(i);
                if *ptr == p0 && *ptr.add(1) == p1 {
                    results.push(i);
                }
                i += 1;
            }
        }

        results
    }

    #[inline]
    fn search_len3(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let text_len = text.len();

        if text_len < 3 {
            return results;
        }

        let p0 = unsafe { *pattern.as_ptr() };
        let p1 = unsafe { *pattern.as_ptr().add(1) };
        let p2 = unsafe { *pattern.as_ptr().add(2) };
        let text_ptr = text.as_ptr();

        unsafe {
            let mut i = 0;
            let limit = text_len - 2;

            while i < limit {
                let ptr = text_ptr.add(i);
                if *ptr == p0 && *ptr.add(1) == p1 && *ptr.add(2) == p2 {
                    results.push(i);
                }
                i += 1;
            }
        }

        results
    }

    #[inline]
    fn search_len4(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let text_len = text.len();

        if text_len < 4 {
            return results;
        }

        let p0 = unsafe { *pattern.as_ptr() };
        let p1 = unsafe { *pattern.as_ptr().add(1) };
        let p2 = unsafe { *pattern.as_ptr().add(2) };
        let p3 = unsafe { *pattern.as_ptr().add(3) };
        let text_ptr = text.as_ptr();

        unsafe {
            let mut i = 0;
            let limit = text_len - 3;

            while i < limit {
                let ptr = text_ptr.add(i);
                if *ptr == p0
                    && *ptr.add(1) == p1
                    && *ptr.add(2) == p2
                    && *ptr.add(3) == p3
                {
                    results.push(i);
                }
                i += 1;
            }
        }

        results
    }

    #[inline]
    fn search_generic(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let text_len = text.len();
        let pattern_len = pattern.len();
        let text_ptr = text.as_ptr();
        let pattern_ptr = pattern.as_ptr();

        let mut shift = [pattern_len; 256];
        unsafe {
            for i in 0..(pattern_len - 1) {
                let byte_idx = *pattern_ptr.add(i) as usize;
                shift[byte_idx] = pattern_len - 1 - i;
            }
        }

        let first_byte = unsafe { *pattern_ptr };
        let second_byte = unsafe { *pattern_ptr.add(1) };
        let last_byte = unsafe { *pattern_ptr.add(pattern_len - 1) };

        let mut results = Vec::new();
        let limit = text_len - pattern_len;
        let mut pos = 0;

        unsafe {
            while pos <= limit {
                let text_ptr_at_pos = text_ptr.add(pos);
                let text_last = *text_ptr_at_pos.add(pattern_len - 1);

                if text_last == last_byte {
                    let text_first = *text_ptr_at_pos;

                    if text_first == first_byte {
                        let text_second = *text_ptr_at_pos.add(1);
                        if text_second == second_byte {
                            if Self::compare_full(text_ptr_at_pos, pattern_ptr, pattern_len) {
                                results.push(pos);
                            }
                        }
                    }
                }

                let shift_amount = shift[text_last as usize];
                pos = pos.saturating_add(shift_amount).max(pos + 1);
            }
        }

        results
    }

    #[inline]
    unsafe fn compare_full(
        text_ptr: *const u8,
        pattern_ptr: *const u8,
        len: usize,
    ) -> bool {
        let mut j = 0;

        let unroll_limit = len.saturating_sub(8);
        while j < unroll_limit {
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
