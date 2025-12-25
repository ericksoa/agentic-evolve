# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection with Rust benchmarks.

## Usage

```
/evolve [target] [goal]
```

Examples:
- `/evolve` - Evolve the string search algorithm in the showcase
- `/evolve src/sort.rs "optimize for nearly-sorted arrays"`

## Execution

When this skill is invoked, perform evolutionary optimization:

### Step 1: Setup

1. Identify the target code (default: `showcase/string-search/initial_program.rs`)
2. Read the current implementation
3. Run the evaluator to establish baseline fitness:
   ```bash
   cd showcase/string-search && python3 evaluator.py initial_program.rs
   ```

### Step 2: Evolution Loop

For each generation (run 3-5 generations):

#### 2a. Generate Mutations (PARALLEL)

Spawn 8-16 mutation agents in parallel using the Task tool. Each agent should:
- Receive the current best code
- Apply a specific mutation strategy
- Return improved code

Mutation strategies to distribute across agents:
- **tweak**: Small micro-optimizations (cache lengths, reorder conditions)
- **restructure**: Different algorithmic approach
- **specialize**: Optimize for common cases
- **vectorize**: SIMD-friendly patterns
- **memoize**: Add caching/lookup tables
- **unroll**: Loop unrolling optimizations
- **alien**: Radically different approach

Example agent prompt:
```
You are an algorithm optimizer. Improve this Rust string search for SPEED.

Current code:
[CODE HERE]

Strategy: [STRATEGY]

Requirements:
- Must implement StringSearch trait
- Must return Vec<usize> of match positions
- Must handle edge cases (empty pattern, overlapping matches)
- Focus on PERFORMANCE

Return ONLY the improved Rust code, no explanations.
```

#### 2b. Evaluate Each Mutation

For each generated mutation:
1. Write it to a temp file
2. Run the evaluator:
   ```bash
   cd showcase/string-search && python3 evaluator.py /path/to/mutation.rs
   ```
3. Parse the JSON result
4. Track: score, searches_per_second, vs_best_baseline

#### 2c. Selection

- Keep the top 3-5 mutations by score
- Use them as parents for the next generation
- Report progress to user

### Step 3: Finalize

1. Take the champion (highest scoring mutation)
2. Write it to `showcase/string-search/rust/src/evolved.rs`
3. Run final benchmark to confirm improvement
4. Report results with before/after comparison

## Output Format

After each generation, report:
```
Generation N:
  Mutations tested: X
  Best fitness: Y.YYY
  Best speed: Z,ZZZ searches/sec
  Improvement vs baseline: +X.X%
```

Final report:
```
Evolution Complete!

Baseline:  X,XXX searches/sec
Champion:  Y,YYY searches/sec
Improvement: +Z.Z%

Champion written to: showcase/string-search/rust/src/evolved.rs
```

## Key Files

- `showcase/string-search/initial_program.rs` - Starting point
- `showcase/string-search/evaluator.py` - Fitness evaluation
- `showcase/string-search/rust/src/evolved.rs` - Where mutations are tested
- `showcase/string-search/rust/src/baselines.rs` - Algorithms to beat
