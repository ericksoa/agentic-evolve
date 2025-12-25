use crate::StringSearch;

/// Two-Way string matching algorithm
pub struct EvolvedSearch {}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self {}
    }

    fn maximal_suffix(pattern: &[u8], reverse: bool) -> (usize, usize) {
        let m = pattern.len();
        let mut pos = 0;
        let mut period = 1;
        let mut i = 1;
        let mut j = 0;

        while i + j < m {
            let a = if reverse {
                pattern[m - 1 - (pos + j)]
            } else {
                pattern[pos + j]
            };
            let b = if reverse {
                pattern[m - 1 - (i + j)]
            } else {
                pattern[i + j]
            };

            if a == b {
                if j + 1 == period {
                    i += period;
                    j = 0;
                } else {
                    j += 1;
                }
            } else if a < b {
                i += j + 1;
                j = 0;
                period = i - pos;
            } else {
                pos = i;
                i = pos + 1;
                j = 0;
                period = 1;
            }
        }

        (pos, period)
    }

    fn critical_factorization(pattern: &[u8]) -> (usize, usize) {
        let (pos1, period1) = Self::maximal_suffix(pattern, false);
        let (pos2, period2) = Self::maximal_suffix(pattern, true);

        let pos2 = pattern.len() - 1 - pos2;

        if pos1 > pos2 {
            (pos1, period1)
        } else {
            (pos2, period2)
        }
    }

    fn is_periodic(pattern: &[u8], period: usize) -> bool {
        let m = pattern.len();
        for i in period..m {
            if pattern[i] != pattern[i - period] {
                return false;
            }
        }
        true
    }

    fn two_way_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();
        let mut results = Vec::new();

        let (critical_pos, period) = Self::critical_factorization(pattern);
        let is_periodic = Self::is_periodic(pattern, period);

        let mut i = 0;
        let mut memory = 0;

        while i <= n - m {
            let mut j = critical_pos.max(memory);

            while j < m && pattern[j] == text[i + j] {
                j += 1;
            }

            if j == m {
                let mut j = critical_pos;
                while j > memory && pattern[j - 1] == text[i + j - 1] {
                    j -= 1;
                }

                if j == memory {
                    results.push(i);
                    if is_periodic {
                        i += period;
                        memory = m - period;
                    } else {
                        i += 1;
                        memory = 0;
                    }
                } else {
                    i += 1;
                    memory = 0;
                }
            } else {
                if is_periodic {
                    i += critical_pos + 1;
                } else {
                    i += j.saturating_sub(critical_pos).max(1);
                }
                memory = 0;
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

        if m == 0 {
            return Vec::new();
        }
        if m > n {
            return Vec::new();
        }
        if m == 1 {
            let target = pattern[0];
            return text
                .iter()
                .enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        Self::two_way_search(text, pattern)
    }
}
