# /evolve Skill Improvement Plan

This document tracks the implementation of 8 improvements to the `/evolve` skill, prioritized by impact and feasibility.

## Current Status: All Improvements Implemented

**Last Updated:** 2025-12-27
**Champion Result:** 16.2% improvement over FunSearch on Weibull 5k (0.5735% vs 0.6842%)
**Skill Updated:** `.claude/commands/evolve.md`

---

## Improvement Queue

### 1. Generalization Testing [PRIORITY: HIGH] [STATUS: COMPLETE]

**Problem:** Current evolution only tests on Weibull 5k. Unknown if improvements generalize.

**Implementation:**
- Add multiple benchmark distributions to skill.md requirements:
  - Weibull (current)
  - Uniform distribution
  - Normal/Gaussian distribution
  - Bimodal distribution
  - Power-law/Pareto distribution
- Require candidates beat baseline on majority of distributions
- Report per-distribution performance in generations.jsonl

**Files to modify:**
- `skill.md` - Add generalization requirements to evaluation contract
- `.evolve/<problem>/rust/src/benchmark.rs` - Template for multi-distribution testing

**Success criteria:** Champion must improve on ≥3/5 distributions to be accepted

---

### 2. Crossover Between Candidates [PRIORITY: HIGH] [STATUS: COMPLETE]

**Problem:** Gen2+ crossover prompts exist but weren't systematically used.

**Implementation:**
- Add explicit crossover phase to evolution loop in skill.md
- Define crossover operator: combine innovations from 2 parents
- Track parent lineage in candidate records
- Allocate 50% of Gen2+ agents to crossover, 50% to mutation

**Files to modify:**
- `skill.md` - Strengthen crossover phase requirements
- Add crossover tracking to generations.jsonl schema

**Success criteria:** Every Gen2+ generation has ≥3 crossover candidates

---

### 3. Population Diversity Tracking [PRIORITY: MEDIUM] [STATUS: COMPLETE]

**Problem:** Population can converge to similar solutions, losing exploration.

**Implementation:**
- Define algorithm_family taxonomy in skill.md
- Track diversity metric: count of distinct families in population
- Force exploration if diversity < 2 families
- Add diversity score to generation logs

**Files to modify:**
- `skill.md` - Add diversity requirements
- `evolution.json` schema - Add diversity metrics

**Success criteria:** Population maintains ≥2 distinct algorithm families

---

### 4. Automated Baseline Discovery [PRIORITY: MEDIUM] [STATUS: COMPLETE]

**Problem:** Manually finding published baselines is error-prone.

**Implementation:**
- Add WebSearch step to find published results for problem domain
- Parse common benchmark formats (papers, GitHub repos)
- Store discovered baselines in evolution.json
- Require improvement vs published state-of-art, not just naive

**Files to modify:**
- `skill.md` - Add baseline discovery to Step 0-pre

**Success criteria:** Automatically find and cite published baselines

---

### 5. Token Budget Display [PRIORITY: LOW] [STATUS: COMPLETE]

**Problem:** No visibility into token usage during evolution.

**Implementation:**
- Estimate tokens per generation based on agent count
- Display running total and remaining budget
- Warn when approaching limit

**Files to modify:**
- `skill.md` - Add token tracking requirements

**Success criteria:** Each generation shows "Tokens: X/Y used (Z%)"

---

### 6. Statistical Confidence Intervals [PRIORITY: LOW] [STATUS: COMPLETE]

**Problem:** Single-run results may be noise. Need confidence bounds.

**Implementation:**
- Run each evaluation N times (default N=5)
- Report mean ± std for all metrics
- Only accept if improvement > 2σ
- Store all runs in generations.jsonl

**Files to modify:**
- `skill.md` - Add statistical requirements
- `evaluation.json` schema - Add per-run data

**Success criteria:** All reported metrics include confidence intervals

---

### 7. Metal GPU Parallel Evaluation [PRIORITY: LOW] [STATUS: COMPLETE]

**Problem:** Sequential evaluation is slow. Apple Silicon has GPU.

**Implementation:**
- Add metal-rs dependency for GPU compute
- Parallelize candidate evaluation across GPU cores
- Batch multiple candidates per evaluation round
- Measure speedup vs sequential

**Files to modify:**
- `skill.md` - Add optional GPU acceleration
- Template Cargo.toml - Add metal-rs dependency

**Success criteria:** ≥4x speedup on M-series chips

---

### 8. GPU-Accelerated Benchmarks [PRIORITY: LOW] [STATUS: COMPLETE]

**Problem:** Some algorithms (sorting, matrix ops) can be GPU-accelerated.

**Implementation:**
- Detect problem types amenable to GPU
- Generate Metal compute shader variants
- Compare CPU vs GPU implementations
- Track which approach wins per problem size

**Files to modify:**
- `skill.md` - Add GPU algorithm evolution
- Template for Metal compute shaders

**Success criteria:** Discover GPU-optimal algorithms for applicable problems

---

## Implementation Log

### 2025-12-27: All 8 Improvements Implemented

**Changes to `.claude/commands/evolve.md`:**

1. **Generalization Testing** (lines 37-118)
   - Added requirement #5 to Evaluation Contract
   - New "Generalization Requirements" section with distribution tables
   - Promotion gate requiring wins on ≥3/5 distributions
   - Overfitting detection flags

2. **Crossover Requirements** (lines 796-853)
   - Made crossover MANDATORY in Gen2+
   - Added minimum crossover count (≥3 per generation)
   - Added diversity preference for parent pairing
   - Added crossover logging requirements with parent tracking

3. **Population Diversity Tracking** (lines 245-336)
   - Added diversity tracking JSON schema
   - Shannon entropy-based diversity score
   - Diversity enforcement rules (family caps)
   - Diversity-aware selection algorithm
   - Low diversity alert with automatic forced exploration

4. **Automated Baseline Discovery** (lines 595-674)
   - New Step 0-pre sections (a-e)
   - Search queries for published baselines
   - Priority ordering for baseline sources
   - Baseline verification protocol
   - Handling for novel benchmarks

5. **Token Budget Display** (lines 927-977)
   - Per-generation budget display box
   - Token tracking in evolution.json
   - Budget warnings at thresholds (10%, 25%, 50%)

6. **Statistical Confidence Intervals** (lines 143-220)
   - Added acceptance criterion #6
   - New "Statistical Rigor Requirements" section
   - Multiple runs with mean/std/CI reporting
   - Statistical significance testing
   - Guidance on when to skip (deterministic benchmarks)

7. **Metal GPU Parallel Evaluation** (lines 1430-1511)
   - New "GPU Acceleration" section
   - Decision table for when to use GPU
   - Metal setup for Cargo.toml
   - Parallel candidate evaluation code
   - Expected speedup table
   - Fallback behavior

8. **GPU-Accelerated Algorithms** (lines 1515-1579)
   - Applicable problems table
   - Metal shader evolution template
   - Hybrid CPU/GPU evolution pattern
   - GPU evolution tracking schema

---

## Notes

- Improvements #1-4 are high-impact, low-risk changes to skill.md
- Improvements #5-6 are quality-of-life enhancements
- Improvements #7-8 require significant new infrastructure (Metal integration)
- Implement in order, testing each before moving to next
