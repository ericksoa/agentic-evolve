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
        for (i, &byte) in pattern.iter().enumerate().take(m - 1) {
            shift[byte as usize] = m - 1 - i;
        }

        let mut results = Vec::new();
        let mut i = 0;
        let last_pattern_byte = pattern[m - 1];
        let first_pattern_byte = pattern[0];

        while i <= n - m {
            let last_text_byte = text[i + m - 1];
            if last_text_byte == last_pattern_byte {
                if text[i] == first_pattern_byte {
                    let mut matched = true;

                    if m >= 10 {
                        let mut j = 1;
                        let end = m - 1;
                        let unroll_end = end - (end - 1) % 8;

                        while j < unroll_end {
                            if text[i + j] != pattern[j] ||
                               text[i + j + 1] != pattern[j + 1] ||
                               text[i + j + 2] != pattern[j + 2] ||
                               text[i + j + 3] != pattern[j + 3] ||
                               text[i + j + 4] != pattern[j + 4] ||
                               text[i + j + 5] != pattern[j + 5] ||
                               text[i + j + 6] != pattern[j + 6] ||
                               text[i + j + 7] != pattern[j + 7] {
                                matched = false;
                                break;
                            }
                            j += 8;
                        }

                        if matched {
                            while j < end {
                                if text[i + j] != pattern[j] {
                                    matched = false;
                                    break;
                                }
                                j += 1;
                            }
                        }
                    } else {
                        let mut j = 1;
                        let end = m - 1;
                        let unroll_end = end - (end - 1) % 4;

                        while j < unroll_end {
                            if text[i + j] != pattern[j] ||
                               text[i + j + 1] != pattern[j + 1] ||
                               text[i + j + 2] != pattern[j + 2] ||
                               text[i + j + 3] != pattern[j + 3] {
                                matched = false;
                                break;
                            }
                            j += 4;
                        }

                        if matched {
                            while j < end {
                                if text[i + j] != pattern[j] {
                                    matched = false;
                                    break;
                                }
                                j += 1;
                            }
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
