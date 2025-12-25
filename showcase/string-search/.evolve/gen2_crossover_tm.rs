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

        if m == 0 { return Vec::new(); }
        if m > n { return Vec::new(); }
        if m == 1 {
            let target = pattern[0];
            return text.iter().enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        unsafe {
            let text_ptr = text.as_ptr();
            let pattern_ptr = pattern.as_ptr();
            let mut shift = [m; 256];

            for i in 0..(m - 1) {
                let byte = *pattern_ptr.add(i) as usize;
                shift[byte] = m - 1 - i;
            }

            let mut shift2 = [m; 256];
            if m >= 2 {
                for i in 0..(m - 2) {
                    let byte = *pattern_ptr.add(i) as usize;
                    shift2[byte] = m - 2 - i;
                }
            }

            let last_pattern_byte = *pattern_ptr.add(m - 1);
            let first_pattern_byte = *pattern_ptr;
            let second_last_byte = if m >= 2 { *pattern_ptr.add(m - 2) } else { 0 };

            let mut char_bitmap = [0u64; 4];
            for i in 0..m {
                let byte = *pattern_ptr.add(i) as usize;
                let idx = byte / 64;
                let bit = byte % 64;
                char_bitmap[idx] |= 1u64 << bit;
            }

            let mut results = Vec::new();
            let mut i = 0;
            let end = n - m;

            while i <= end {
                let last_text_byte = *text_ptr.add(i + m - 1);

                let idx = (last_text_byte as usize) / 64;
                let bit = (last_text_byte as usize) % 64;
                if (char_bitmap[idx] & (1u64 << bit)) == 0 {
                    i += m;
                    continue;
                }

                if last_text_byte == last_pattern_byte {
                    if m >= 2 {
                        let second_last_text = *text_ptr.add(i + m - 2);
                        if second_last_text != second_last_byte {
                            i += shift2[second_last_text as usize].max(1);
                            continue;
                        }
                    }

                    if *text_ptr.add(i) == first_pattern_byte {
                        let mut matched = true;
                        for j in 1..(m - 1) {
                            if *text_ptr.add(i + j) != *pattern_ptr.add(j) {
                                matched = false;
                                break;
                            }
                        }
                        if matched { results.push(i); }
                    }
                }
                i += shift[last_text_byte as usize].max(1);
            }
            results
        }
    }
}
