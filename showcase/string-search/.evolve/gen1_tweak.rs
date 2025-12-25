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

        let mut shift = [m; 256];
        for i in 0..(m - 1) {
            shift[unsafe { *pattern.get_unchecked(i) } as usize] = m - 1 - i;
        }

        let mut results = Vec::new();
        let mut i = 0;
        let last_pattern_byte = unsafe { *pattern.get_unchecked(m - 1) };
        let first_pattern_byte = unsafe { *pattern.get_unchecked(0) };
        let end = n - m;
        let m_minus_1 = m - 1;

        while i <= end {
            let last_text_byte = unsafe { *text.get_unchecked(i + m_minus_1) };
            if last_text_byte == last_pattern_byte && unsafe { *text.get_unchecked(i) } == first_pattern_byte {
                let mut matched = true;
                for j in 1..m_minus_1 {
                    if unsafe { *text.get_unchecked(i + j) != *pattern.get_unchecked(j) } {
                        matched = false;
                        break;
                    }
                }
                if matched { results.push(i); }
            }
            i += shift[last_text_byte as usize];
        }
        results
    }
}
