use crate::StringSearch;

pub struct EvolvedSearch {
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
    }

    #[inline(always)]
    fn compute_shift_table(pattern: &[u8]) -> [usize; 256] {
        let m = pattern.len();
        let mut shift = [m; 256];
        let m_minus_1 = m - 1;

        for i in 0..m_minus_1 {
            shift[pattern[i] as usize] = m_minus_1 - i;
        }
        shift
    }

    #[inline(always)]
    fn quick_check(text: &[u8], pattern: &[u8], pos: usize, m: usize) -> bool {
        unsafe {
            *text.get_unchecked(pos + m - 1) == *pattern.get_unchecked(m - 1) &&
            *text.get_unchecked(pos) == *pattern.get_unchecked(0)
        }
    }

    #[inline(always)]
    fn full_match(text: &[u8], pattern: &[u8], pos: usize, m: usize) -> bool {
        unsafe {
            let text_ptr = text.as_ptr().add(pos);
            let pattern_ptr = pattern.as_ptr();

            for j in 1..(m - 1) {
                if *text_ptr.add(j) != *pattern_ptr.add(j) {
                    return false;
                }
            }
            true
        }
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
            let mut results = Vec::new();
            for (i, &b) in text.iter().enumerate() {
                if b == target {
                    results.push(i);
                }
            }
            return results;
        }

        let shift = Self::compute_shift_table(pattern);
        let mut results = Vec::with_capacity((n / m).min(256));
        let end = n - m;
        let mut i = 0;

        while i <= end {
            if Self::quick_check(text, pattern, i, m) {
                if Self::full_match(text, pattern, i, m) {
                    results.push(i);
                }
            }

            let shift_val = unsafe { shift[*text.get_unchecked(i + m - 1) as usize] };
            i += shift_val;
        }

        results
    }
}
