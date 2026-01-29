# Phase 1 Showcase Options

We need a problem that exercises both new agents:

**Debugger Agent** - needs frequent mutation failures to diagnose
**Plateau Breaker** - needs natural local optima to escape

## Selection Criteria

| Criterion | Weight | Why |
|-----------|--------|-----|
| High failure rate | High | Debugger needs failures to analyze |
| Natural plateaus | High | Plateau Breaker needs stalls to detect |
| Fast evaluation | Medium | More generations = more data |
| Clear fitness metric | Medium | Easy to measure improvement |
| Interesting problem | Low | Nice to have |

---

## Option 1: Regex Golf

**Problem**: Generate the shortest regex that matches all "good" strings and rejects all "bad" strings.

**Example**:
```
Good: ["foo", "food", "fool", "foot"]
Bad: ["bar", "baz", "far", "foe"]
Answer: "foo[dlt]?" or "fo{2}[dlt]?"
```

**Why it's good for Phase 1**:

| Agent | Exercise |
|-------|----------|
| Debugger | Regex syntax is fragile - one wrong character = crash. `re.compile()` throws on invalid patterns. High failure rate guaranteed. |
| Plateau Breaker | Easy to get stuck on "good enough" regex. Escaping local minima requires structural changes (character classes → groups → lookahead). |

**Fitness**: `1000 - len(regex)` (smaller = better), with penalty for incorrect matches

**Pros**:
- Very high mutation failure rate (exercises Debugger heavily)
- Clear local optima (exercises Plateau Breaker)
- Fast evaluation (regex matching is instant)
- Small code size (easy to analyze)

**Cons**:
- Somewhat artificial problem
- Regex expertise helps interpret results

**Estimated generations to plateau**: 5-8

---

## Option 2: Expression Simplifier

**Problem**: Simplify mathematical expressions to shortest equivalent form.

**Example**:
```
Input: "x + x + x"
Output: "3*x"

Input: "x * 1 + 0"
Output: "x"

Input: "(a + b) * (a + b)"
Output: "(a+b)**2"
```

**Why it's good for Phase 1**:

| Agent | Exercise |
|-------|----------|
| Debugger | AST manipulation breaks easily - wrong node types, missing cases, infinite recursion. Many failure modes. |
| Plateau Breaker | Easy to get stuck after obvious simplifications (x+0=x, x*1=x). Breaking plateau requires new algebraic rules. |

**Fitness**: Reduction in expression length while maintaining correctness

**Pros**:
- Natural multi-stage plateaus (each rule set is a local optimum)
- Rich failure modes (type errors, recursion errors, wrong simplifications)
- Mathematically interesting
- Easy to verify correctness

**Cons**:
- Need test suite of expressions
- More complex to set up

**Estimated generations to plateau**: 8-12

---

## Option 3: Huffman-style Compression

**Problem**: Evolve a compression algorithm for a specific corpus.

**Example**:
```
Corpus: English text
Baseline: 8 bits/char
Goal: < 5 bits/char
```

**Why it's good for Phase 1**:

| Agent | Exercise |
|-------|----------|
| Debugger | Bit manipulation is error-prone. Off-by-one in bit shifts, buffer overflows, encoding/decoding mismatches. |
| Plateau Breaker | Clear plateaus at each compression technique level (basic → run-length → dictionary → entropy). |

**Fitness**: Compression ratio (smaller output = higher fitness)

**Pros**:
- Real-world applicable
- Clear improvement trajectory
- Multiple algorithm paradigms to explore

**Cons**:
- Slower evaluation (need to compress/decompress)
- Harder to verify correctness
- More complex implementation

**Estimated generations to plateau**: 10-15

---

## Option 4: Scheduling Optimizer

**Problem**: Schedule N tasks with dependencies and constraints to minimize total time.

**Example**:
```
Tasks: A(2h), B(3h), C(1h), D(4h)
Dependencies: A→C, B→C, C→D
Workers: 2
Optimal: 7 hours (A,B parallel, then C, then D)
```

**Why it's good for Phase 1**:

| Agent | Exercise |
|-------|----------|
| Debugger | Constraint violations cause crashes - invalid schedules, dependency cycles, resource overallocation. |
| Plateau Breaker | Greedy solutions plateau fast. Escaping requires different paradigms (genetic, constraint propagation, ILP). |

**Fitness**: -makespan (lower is better)

**Pros**:
- Practical problem
- Clear local vs global optima
- Rich constraint violations

**Cons**:
- Need to generate test cases
- Optimal solution may be NP-hard to verify

**Estimated generations to plateau**: 6-10

---

## Option 5: Bit Manipulation Tricks

**Problem**: Implement common bit operations in fewest instructions.

**Example**:
```
Task: Count set bits in integer
Naive: Loop through all bits (32 ops)
Clever: Brian Kernighan's algorithm (ops = popcount)
Best: SWAR/parallel counting (5-6 ops)
```

**Why it's good for Phase 1**:

| Agent | Exercise |
|-------|----------|
| Debugger | Bit operations are fragile. Wrong shift direction, missing mask, overflow. Very easy to break. |
| Plateau Breaker | Each algorithm family is a distinct optimum. Breaking plateau requires paradigm shift. |

**Fitness**: Speed (ops/sec) or instruction count

**Pros**:
- Classic optimization problem
- Very fast evaluation
- Clear improvement levels
- Small code to analyze

**Cons**:
- Narrow domain
- May hit hardware limits quickly

**Estimated generations to plateau**: 4-7

---

## Recommendation

**For maximum Phase 1 agent exercise, I recommend: Regex Golf**

Reasons:
1. **Highest failure rate** - regex syntax is unforgiving, every mutation risks a crash
2. **Fast iteration** - regex matching is instant, can run many generations quickly
3. **Clear plateaus** - easy to get "good enough" regex, hard to escape
4. **Small code** - easy to analyze diffs, clear root causes
5. **Self-contained** - no external dependencies, just `re` module

**Second choice: Expression Simplifier**
- More intellectually interesting
- Better demonstrates multi-stage plateaus
- Slightly lower failure rate but richer failures

---

## Your Choice

Which showcase would you like me to build?

1. **Regex Golf** - Maximum failure rate, fast iteration
2. **Expression Simplifier** - Interesting plateaus, algebraic challenge
3. **Scheduling Optimizer** - Practical problem, constraint violations
4. **Bit Manipulation** - Fast, clear paradigm shifts
5. **Other** - Suggest your own

