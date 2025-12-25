use crate::StringSearch;

/// Bloom Filter Cascade Search Algorithm
#[derive(Clone)]
pub struct EvolvedSearch {
    filter_2gram: [u64; 4],
    filter_4gram: [u64; 4],
    filter_8gram: [u64; 4],
    pattern_fingerprint: u64,
    pattern_len: usize,
    byte_signature: [u8; 8],
    short_pattern: [u8; 32],
}

impl Default for EvolvedSearch {
    fn default() -> Self {
        Self {
            filter_2gram: [0; 4],
            filter_4gram: [0; 4],
            filter_8gram: [0; 4],
            pattern_fingerprint: 0,
            pattern_len: 0,
            byte_signature: [0; 8],
            short_pattern: [0; 32],
        }
    }
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn hash1(bytes: &[u8]) -> u8 {
        let mut h: u32 = 2166136261;
        for &b in bytes {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
        (h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24)) as u8
    }

    #[inline(always)]
    fn hash2(bytes: &[u8]) -> u8 {
        let mut h: u32 = 0x811c9dc5;
        for &b in bytes {
            h = h.wrapping_add(b as u32);
            h = h.wrapping_mul(0x01000193);
            h ^= h >> 13;
        }
        (h ^ (h >> 8)) as u8
    }

    #[inline(always)]
    fn bloom_set(filter: &mut [u64; 4], h1: u8, h2: u8) {
        let idx1 = (h1 as usize) >> 6;
        let bit1 = h1 & 63;
        let idx2 = (h2 as usize) >> 6;
        let bit2 = h2 & 63;
        filter[idx1] |= 1u64 << bit1;
        filter[idx2] |= 1u64 << bit2;
    }

    #[inline(always)]
    fn bloom_check(filter: &[u64; 4], h1: u8, h2: u8) -> bool {
        let idx1 = (h1 as usize) >> 6;
        let bit1 = h1 & 63;
        let idx2 = (h2 as usize) >> 6;
        let bit2 = h2 & 63;
        (filter[idx1] & (1u64 << bit1)) != 0 && (filter[idx2] & (1u64 << bit2)) != 0
    }

    #[inline(always)]
    fn fingerprint(bytes: &[u8]) -> u64 {
        let mut fp: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            fp ^= b as u64;
            fp = fp.wrapping_mul(0x100000001b3);
        }
        fp
    }

    fn preprocess(&mut self, pattern: &[u8]) {
        self.pattern_len = pattern.len();

        self.filter_2gram = [0; 4];
        self.filter_4gram = [0; 4];
        self.filter_8gram = [0; 4];
        self.byte_signature = [0; 8];

        if pattern.is_empty() {
            return;
        }

        let copy_len = pattern.len().min(32);
        self.short_pattern[..copy_len].copy_from_slice(&pattern[..copy_len]);

        for (i, &b) in pattern.iter().enumerate() {
            self.byte_signature[i & 7] ^= b;
        }

        if pattern.len() >= 2 {
            for window in pattern.windows(2) {
                let h1 = Self::hash1(window);
                let h2 = Self::hash2(window);
                Self::bloom_set(&mut self.filter_2gram, h1, h2);
            }
        }

        if pattern.len() >= 4 {
            for window in pattern.windows(4) {
                let h1 = Self::hash1(window);
                let h2 = Self::hash2(window);
                Self::bloom_set(&mut self.filter_4gram, h1, h2);
            }
        }

        if pattern.len() >= 8 {
            for window in pattern.windows(8) {
                let h1 = Self::hash1(window);
                let h2 = Self::hash2(window);
                Self::bloom_set(&mut self.filter_8gram, h1, h2);
            }
        }

        self.pattern_fingerprint = Self::fingerprint(pattern);
    }

    #[inline(always)]
    fn signature_matches(&self, text: &[u8], pos: usize, len: usize) -> bool {
        let mut sig = [0u8; 8];
        for i in 0..len {
            sig[i & 7] ^= text[pos + i];
        }
        sig == self.byte_signature
    }

    #[inline(always)]
    fn bloom_cascade_check(&self, text: &[u8], pos: usize) -> bool {
        let len = self.pattern_len;

        if len >= 2 {
            let first = &text[pos..pos + 2];
            if !Self::bloom_check(&self.filter_2gram, Self::hash1(first), Self::hash2(first)) {
                return false;
            }

            let last = &text[pos + len - 2..pos + len];
            if !Self::bloom_check(&self.filter_2gram, Self::hash1(last), Self::hash2(last)) {
                return false;
            }

            if len >= 4 {
                let mid = &text[pos + len / 2 - 1..pos + len / 2 + 1];
                if !Self::bloom_check(&self.filter_2gram, Self::hash1(mid), Self::hash2(mid)) {
                    return false;
                }
            }
        }

        if len >= 8 {
            let first4 = &text[pos..pos + 4];
            if !Self::bloom_check(&self.filter_4gram, Self::hash1(first4), Self::hash2(first4)) {
                return false;
            }

            let last4 = &text[pos + len - 4..pos + len];
            if !Self::bloom_check(&self.filter_4gram, Self::hash1(last4), Self::hash2(last4)) {
                return false;
            }
        }

        if len >= 16 {
            let first8 = &text[pos..pos + 8];
            if !Self::bloom_check(&self.filter_8gram, Self::hash1(first8), Self::hash2(first8)) {
                return false;
            }
        }

        true
    }
}

impl StringSearch for EvolvedSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();

        if pattern.is_empty() {
            return results;
        }

        if pattern.len() > text.len() {
            return results;
        }

        let n = text.len();
        let m = pattern.len();

        if m <= 3 {
            for i in 0..=n - m {
                if &text[i..i + m] == pattern {
                    results.push(i);
                }
            }
            return results;
        }

        let mut searcher = self.clone();
        searcher.preprocess(pattern);

        let mut pos = 0;

        while pos <= n - m {
            if text[pos] != pattern[0] {
                pos += 1;
                continue;
            }

            if text[pos + m - 1] != pattern[m - 1] {
                pos += 1;
                continue;
            }

            if searcher.bloom_cascade_check(text, pos) {
                if searcher.signature_matches(text, pos, m) {
                    let candidate_fp = Self::fingerprint(&text[pos..pos + m]);

                    if candidate_fp == searcher.pattern_fingerprint {
                        if &text[pos..pos + m] == pattern {
                            results.push(pos);
                        }
                    }
                }
            }

            pos += 1;
        }

        results
    }
}
