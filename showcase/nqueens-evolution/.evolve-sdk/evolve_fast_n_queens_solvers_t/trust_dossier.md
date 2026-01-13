# Trust Dossier

> **Problem:** Evolve fast N-Queens solvers that generalize across board sizes
> **Generated:** 2026-01-12 19:21:36

---

## Summary

| Metric | Value |
|--------|-------|
| Total Evaluations | 10 |
| Accepted | 6 |
| Challenged | 0 |
| Rejected | 4 |
| Human Escalations | 4 |
| Human Accepts | 0 |
| Human Rejects | 4 |

## Human Escalations

| Candidate | Fitness | Original Trust | Decision | Adjusted Trust |
|-----------|---------|----------------|----------|----------------|
| gen6a.py | 24336.4510 | 0.00 | reject | - |
| gen6c.py | 44634.5375 | 0.00 | reject | - |
| gen9b.py | 30214.3333 | 0.00 | reject | - |
| gen10c.py | 29581.2460 | 0.00 | reject | - |

## Trust History

### Gen 1: gen1a.py

- **Fitness:** 22025.9103
- **Trust Score:** 0.85
- **Recommendation:** ACCEPT
- **Flags:**
  - Suspicious jump: 48.3%
  - large_fitness_jump_48_percent
  - single_generation_improvement
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 22025.9103
- **Analysis:** The claimed 48.3% improvement is significant but explainable. The mutation makes three legitimate optimizations: (1) Inlines count_available() and get_available() into a single pass, eliminating dupli...

### Gen 5: gen5a.py

- **Fitness:** 21318.1659
- **Trust Score:** 0.85
- **Recommendation:** ACCEPT
- **Flags:**
  - 13.9% jump near suspicious threshold
  - micro-optimization with marginal theoretical benefit
  - python-specific improvement may not generalize to other interpreters
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 21318.1659
- **Analysis:** The candidate replaces inline arithmetic (base_d1 - col, row + col) with precomputed 2D lookup tables for diagonal indices. This is a legitimate memory-for-speed tradeoff optimization. The algorithm s...

### Gen 5: gen5x.py

- **Fitness:** 30035.6671
- **Trust Score:** 0.50
- **Recommendation:** ACCEPT
- **Flags:**
  - Could not parse adversary response
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 30035.6671
- **Analysis:** Adversary response parsing failed, applying default trust penalty

### Gen 6: gen6a.py

- **Fitness:** 24336.4510
- **Trust Score:** 0.00
- **Recommendation:** REJECT
- **Flags:**
  - Suspicious jump: 30.0%
  - Hardcoded lookup table for n <= 8
  - 30% fitness jump exceeds normal threshold
  - Precomputation optimizes for benchmark rather than algorithm
  - 4 test cases likely include small n values that hit lookup
  - Could not parse escalation response
  - human_confirmed_rejection: interrupted
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 24336.451
- **Analysis:** The candidate introduces precomputed solutions for n=0 through n=8, which is a legitimate but benchmark-gaming optimization. The hardcoded values are mathematically correct, and the solution still wor...

### Gen 6: gen6c.py

- **Fitness:** 44634.5375
- **Trust Score:** 0.00
- **Recommendation:** REJECT
- **Flags:**
  - Suspicious jump: 164.4%
  - fitness_jump_164_percent
  - only_4_test_cases
  - subtle_bit_math_in_diagonal_tracking
  - Could not parse escalation response
  - human_confirmed_rejection: interrupted
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 44634.5375
- **Analysis:** The mutation introduces a legitimate algorithmic improvement: converting O(n) per-row availability calculation to O(1) bitwise operations in the MRV solver. The math for diagonal tracking via shifted ...

### Gen 6: gen6x.py

- **Fitness:** 22674.1427
- **Trust Score:** 0.90
- **Recommendation:** ACCEPT
- **Flags:**
  - 21% jump is relative to trust-adjusted champion (only ~2.9% vs original fitness)
  - crossover_combining_proven_techniques
  - small_n_bitwise_from_gen5b
  - large_n_mrv_with_gen5a_lookups
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 22674.1427
- **Analysis:** The gen6x crossover legitimately combines proven techniques from three high-performing parents. The claimed 21.1% improvement is misleading - it compares against gen1a's trust-adjusted fitness (18722)...

### Gen 8: gen8b.py

- **Fitness:** 22064.8900
- **Trust Score:** 0.85
- **Recommendation:** ACCEPT
- **Flags:**
  - Suspicious jump: 63.1%
  - large_fitness_jump_63%
  - algorithmic_threshold_may_be_tuned_to_test_cases
  - major_algorithmic_change_not_incremental
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 30539.2235
- **Analysis:** The candidate introduces a legitimate and well-known optimization: using bitwise backtracking for small N-Queens instead of MRV. The bitwise approach (using bit & -bit for lowest-set-bit extraction, b...

### Gen 9: gen9b.py

- **Fitness:** 30214.3333
- **Trust Score:** 0.00
- **Recommendation:** REJECT
- **Flags:**
  - Suspicious jump: 61.1%
  - fitness_jump_61_percent_implausible_for_micro_optimization
  - measurement_time_too_short_1.32ms_high_variance
  - optimization_only_affects_n_leq_12_cases_but_claims_affect_all
  - mrv_code_path_unchanged_yet_overall_61_percent_improvement
  - Could not parse escalation response
  - human_confirmed_rejection: interrupted
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 30214.3333
- **Analysis:** The code change is legitimate: replacing bit.bit_length()-1 with a precomputed dictionary lookup and local variable binding is a valid Python micro-optimization. HOWEVER, the claimed 61% improvement i...

### Gen 10: gen10b.py

- **Fitness:** 21195.3310
- **Trust Score:** 0.88
- **Recommendation:** ACCEPT
- **Flags:**
  - small_evaluation_set
  - cache_hypothesis_unverified
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 21195.331
- **Analysis:** The candidate solution makes a legitimate optimization: replacing 2D lookup tables with inline arithmetic computation for diagonal indices. The mathematical transformation is correct (base_d1 - col eq...

### Gen 10: gen10c.py

- **Fitness:** 29581.2460
- **Trust Score:** 0.00
- **Recommendation:** REJECT
- **Flags:**
  - Suspicious jump: 57.7%
  - Suspicious jump: 57.7% from a micro-optimization
  - Total time SLOWER than parent (1.35ms vs 1.31ms)
  - Overall speed SLOWER than parent (2958 vs 3053 sol/sec)
  - Single test case (n=12) shows anomalous 5-6x speedup (0.10ms vs 0.51ms)
  - Dictionary lookup replacing bit_length() unlikely to yield 57.7% improvement
  - Measurement variance/noise likely explanation
  - Fitness calculation may be dominated by single anomalous test case result
  - Could not parse escalation response
  - human_confirmed_rejection: interrupted
- **Exploit Detection:**
  - output_integrity: PASS
    - Fitness value valid: 29581.246
- **Analysis:** CRITICAL INCONSISTENCY: The candidate gen10c shows SLOWER overall performance than parent gen8b (1.35ms total vs 1.31ms, 2958 sol/sec vs 3053 sol/sec), yet claims 57.7% fitness improvement. The only e...

---

*Generated by [Agentic Evolve](https://github.com/anthropics/agentic-evolve) Trust System*