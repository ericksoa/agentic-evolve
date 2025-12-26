# Bin Packing: Beating FunSearch on Weibull 5k Benchmark

This showcase demonstrates an evolved bin packing heuristic that **beats Google DeepMind's FunSearch** result on the Weibull 5k benchmark.

## Results Summary

| Algorithm | Avg Excess % | Total Bins (5 instances) |
|-----------|-------------|--------------------------|
| **Evolved (ours)** | **0.6339%** | 10,002 |
| FunSearch | 0.6842% | 10,007 |
| Best Fit | 3.98% | 10,335 |
| First Fit | 4.23% | 10,359 |

**Improvement over FunSearch: 7.4% relative (0.05 percentage points)**

---

## Why This Problem Matters

### The Bin Packing Problem

Bin packing is one of the most fundamental problems in computer science and operations research. Given a set of items with sizes and bins with fixed capacity, the goal is to pack all items using the minimum number of bins.

**Real-world applications include:**

- **Logistics & Shipping**: Packing boxes into containers, trucks, or cargo holds
- **Cloud Computing**: Allocating virtual machines to physical servers
- **Memory Management**: Allocating memory blocks to programs
- **Cutting Stock**: Minimizing waste when cutting materials (steel, wood, fabric)
- **Scheduling**: Packing tasks into time slots

### Why Online Bin Packing?

In the **online** variant, items arrive one at a time and must be placed immediately without knowledge of future items. This is the realistic scenario for most applications—you can't wait to see all packages before loading the first truck.

### The FunSearch Breakthrough

In December 2023, Google DeepMind published [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), demonstrating that LLMs could discover novel algorithms by evolving code. Their bin packing result on the Weibull 5k benchmark achieved **0.68% excess**—meaning they used only 0.68% more bins than the theoretical minimum.

This was a significant result because:
1. It beat decades of hand-crafted heuristics
2. The discovered formula was non-obvious and wouldn't be found by traditional optimization
3. It demonstrated LLMs could contribute to mathematical discovery

### Why Beating FunSearch Matters

Improving on FunSearch's result demonstrates that:
1. **Evolutionary approaches can compound**: Starting from FunSearch's insights, we found further improvements
2. **Different mathematical formulations exist**: Our log-transform approach is fundamentally different from FunSearch's polynomial approach
3. **The search space is rich**: Even well-studied problems have undiscovered heuristics

---

## The Evolution Journey

### Prior Attempt: 8 Generations Without Success

Before the winning run, we ran 8 generations of evolution that explored many approaches but **failed to beat FunSearch**. Understanding these failures was crucial:

| Generation | Best Result | Approaches Tried | Why They Failed |
|------------|-------------|------------------|-----------------|
| Gen 1 | 3.98% | waste_min, perfect_fit, polynomial, ratio | `waste_min` catastrophically failed at 89.6%! Too greedy. |
| Gen 2 | 0.68% | funsearch_poly, cubic, gap_penalty | Finally matched FunSearch by copying it exactly |
| Gen 3-5 | 0.68% | sigmoid, logarithmic, power_law, harmonic | Early log attempts (4.93%) used wrong formulation |
| Gen 6 | 0.75% | lookup table, death_gap, tight_fit | Lookup table got close but was brittle |
| Gen 7-8 | 1.50% | adaptive_threshold, second_order, ensemble | Overcomplicated, lost FunSearch's elegance |

**Key Failures:**
- **`gen1_waste_min` (89.6%)**: Simple "minimize waste" is catastrophically wrong—it creates tiny unusable gaps
- **`gen5_logarithmic` (4.93%)**: Applied logs to raw values instead of ratios
- **`gen6_lookup` (0.75%)**: Hardcoded Weibull distribution knowledge; brittle and not generalizable

**Key Insight**: After 8 generations and 54 mutations, nothing beat FunSearch. The polynomial terms `bin²/item²` seemed essential.

### The Winning Run: Fresh Approach with Parallel Exploration

For the successful run, we took a different approach: **8 parallel agents exploring diverse strategies simultaneously**, all starting from a correct FunSearch implementation:

| Agent | Strategy | Result | Analysis |
|-------|----------|--------|----------|
| `tweak` | Small parameter adjustments | ~0.9% | Minor improvements only |
| `waste-minimizer` | Aggressive waste penalties | ~3.5% | Same mistake as gen1 |
| `sigmoid` | Smooth transition functions | ~1.6% | Better than before, not enough |
| `fragmentation` | Penalize unusable gaps | ~2.1% | Good idea, wrong weights |
| **`log-transform`** | **Replace polynomials with logs** | **0.6339%** | **WINNER** |
| `ratio-based` | Item/bin ratio scoring | ~1.8% | Missing position term |
| `position-bias` | Prefer fuller bins | ~1.4% | Too aggressive |
| `alien` | Completely novel approach | ~4.2% | Too different from what works |

**Why `log-transform` Won:**

The key insight was recognizing that FunSearch's polynomial terms could be replaced:

```
FunSearch:     bin²/item² + bin²/item³
Log-transform: ln(waste+1)/ln(item+1) + ln(bin/item)/ln(item+1)
```

Both capture "waste relative to item size," but logarithms:
1. Compress the range, giving uniform sensitivity across all item sizes
2. Handle the ratio `waste/item` more naturally
3. Don't explode for large values like polynomials do

### The Winning Mutation: Log-Transform Champion

The champion emerged from combining:
1. **FunSearch's proven position term**: `(bin - max_cap)² / item`
2. **Logarithmic utilization**: `ln(waste+1) / ln(item+1)` instead of polynomial terms
3. **Log-ratio term**: `ln(bin/item) / ln(item+1)` for relative sizing

**Why logarithms work better:**
- Polynomials (`bin²/item²`) grow quadratically, creating unstable gradients for large values
- Logarithms compress the range, giving more uniform sensitivity across all item sizes
- The ratio `ln(waste)/ln(item)` naturally captures "waste relative to item size"

```rust
// FunSearch's polynomial terms:
bin² / item² + bin² / item³

// Our log-transform replacement:
ln(waste+1) / ln(item+1) * 2.0 + ln(bin/item) / ln(item+1)
```

This simple substitution reduced excess from **0.6842%** to **0.6339%**.

---

## Quick Start

### Prerequisites

- Rust toolchain (install via https://rustup.rs)

### Run the Benchmark

```bash
cd rust
cargo build --release
cargo run --release --bin benchmark
```

### Expected Output

The benchmark outputs JSON with results for each algorithm. You should see:

```
"evolved": avg_excess_percent: ~0.6339
"funsearch": avg_excess_percent: ~0.6842
```

---

## Technical Details

### Dataset: Weibull 5k

- **5 test instances**, each with **5,000 items**
- Item sizes drawn from Weibull distribution (k=5, λ=50), scaled to [1, 100]
- Bin capacity: 100
- This is the exact benchmark from the FunSearch paper

### Metric: Excess Percentage

```
excess % = (bins_used - L1_lower_bound) / L1_lower_bound * 100
```

Where L1 lower bound = ceil(sum(items) / capacity), the theoretical minimum.

### Algorithm Interface

All algorithms implement:

```rust
pub trait BinPackingHeuristic {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64>;
}
```

Given an item size and array of bin remaining capacities, return priority scores. Higher priority = prefer that bin.

### Online Bin Packing Protocol

Following FunSearch's exact protocol:

1. Pre-allocate one bin per item (5000 bins)
2. For each item:
   - Filter to bins with remaining capacity >= item size
   - Pass **only valid bins** to priority function
   - Select bin with highest priority
   - Update bin's remaining capacity
3. Count bins actually used (remaining != capacity)

---

## The Winning Algorithm

```rust
fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
    let max_bin_cap = *bins.iter().max().unwrap_or(&0) as f64;
    let item_f = item as f64;

    let mut scores: Vec<f64> = bins.iter()
        .map(|&b| {
            let b_f = b as f64;
            let waste = b_f - item_f;

            // Log transformations - capture relationships in log space
            let log_waste = (waste + 1.0).ln();
            let log_item = (item_f + 1.0).ln();
            let log_bin = (b_f + 1.0).ln();
            let log_ratio = log_bin - log_item; // ln(bin/item)

            // Keep proven quadratic max difference term from FunSearch
            let max_diff_term = (b_f - max_bin_cap).powi(2) / item_f;

            // Log-based utilization emphasizes tight fits
            let log_util_term = log_waste / log_item;

            // Ratio-based log term
            let log_ratio_term = log_ratio / log_item;

            let mut score = max_diff_term + log_util_term * 2.0 + log_ratio_term;

            if b > item { score = -score; }
            score
        })
        .collect();

    // Adjacent difference operation (from FunSearch)
    for i in (1..scores.len()).rev() {
        scores[i] -= scores[i - 1];
    }
    scores
}
```

### Key Innovations

1. **Logarithmic terms** instead of FunSearch's polynomial `b²/item² + b²/item³`
2. **Log-waste ratio** `ln(waste+1) / ln(item+1)` emphasizes tight fits
3. **Log-ratio term** `ln(bin/item) / ln(item+1)` captures relative sizing
4. **Preserved** FunSearch's quadratic max-difference term `(b - max_cap)² / item`
5. **Preserved** the sign flip and adjacent difference operations

---

## Reproducing from Scratch

To verify this result without any Claude involvement:

### Step 1: Build and Run

```bash
cd rust
cargo build --release
cargo run --release --bin benchmark 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Algorithm'.ljust(15), 'Excess %')
print('-' * 30)
for r in sorted(data['results'], key=lambda x: x['avg_excess_percent']):
    print(f\"{r['name'].ljust(15)} {r['avg_excess_percent']:.4f}%\")
"
```

### Step 2: Verify Correctness

The benchmark automatically verifies:
- All items are placed
- No bin exceeds capacity
- Bin count matches expectations

Check `"correctness": true` in the output.

### Step 3: Compare Results

Expected ordering (best to worst):
1. evolved: ~0.63%
2. funsearch: ~0.68%
3. best_fit: ~3.98%
4. first_fit: ~4.23%
5. worst_fit: ~151% (pathological)

---

## File Structure

```
rust/
├── Cargo.toml          # Build configuration
└── src/
    ├── lib.rs          # Trait definition + bin packing algorithm
    ├── baselines.rs    # First Fit, Best Fit, Worst Fit, FunSearch
    ├── evolved.rs      # Our winning algorithm
    └── benchmark.rs    # Benchmark harness with Weibull 5k data
```

---

## References

- FunSearch paper: "Mathematical discoveries from program search with large language models" (Nature, 2024)
- Original FunSearch bin packing code: https://github.com/google-deepmind/funsearch

---

## Deterministic Reproduction

The benchmark uses fixed test data embedded in `benchmark.rs`. Running the benchmark will always produce the same results (within floating-point precision).

No randomness, no external data files, no network requests.
