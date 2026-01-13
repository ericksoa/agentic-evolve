# Regex Golf Showcase: Phase 1 Agent Demonstration

This showcase demonstrates the value of the Phase 1 agents (Debugger and Plateau Breaker) on a real optimization problem.

## Problem: Star Wars vs Star Trek

**Goal**: Find the shortest regex that matches all Star Wars movie titles and rejects all Star Trek movie titles.

```
Good (match): "A New Hope", "The Empire Strikes Back", "Return of the Jedi",
              "The Phantom Menace", "Attack of the Clones", "Revenge of the Sith"

Bad (reject): "The Motion Picture", "The Wrath of Khan", "The Search for Spock",
              "The Voyage Home", "The Final Frontier", "The Undiscovered Country"
```

## Why This Problem Exercises Phase 1 Agents

### Debugger Agent Exercise
- **Regex syntax is fragile** - one wrong character crashes `re.compile()`
- Common mutations produce invalid patterns:
  - Unclosed brackets: `pattern[`
  - Bad escapes: `pattern\`
  - Invalid ranges: `[z-a]`
  - Unclosed groups: `(pattern`
- **Observed failure rate: 33%** - high enough to demonstrate value

### Plateau Breaker Exercise
- **Local optima are common** - easy to find a "good enough" regex
- Baseline pattern `Hope|Empire|Jedi|Phantom|Clones|Sith` is correct but long (36 chars)
- Incremental mutations (shorten words, add anchors) don't improve it
- **Breakthrough requires paradigm shift** - find distinguishing substrings instead of full words

## Evolution Results

### Run Summary
```
Generations: 4
Baseline fitness: 964 (pattern length: 36)
Final fitness: 977 (pattern length: 23)
Improvement: +13 points (36% shorter regex)
```

### Phase 1 Agent Activity

#### Debugger Agent: 4 Invocations

| Generation | Error | Diagnosis |
|------------|-------|-----------|
| 1c | `unterminated character set at position 36` | Unclosed `[` bracket |
| 3a | `bad character range z-a at position 20` | Backwards range in character class |
| 4a | `bad character range z-a at position 20` | Same error repeated (pattern learning opportunity) |
| 4b | `unterminated character set at position 36` | Unclosed bracket again |

**What the Debugger provides:**
1. **Root cause analysis** - not just "regex error" but "unclosed bracket at position 36"
2. **Failure categorization** - `syntax_error` enables pattern matching across runs
3. **Lessons learned** - "validate regex syntax before applying mutation"
4. **Pattern detection** - 2 failures in gen 4 triggered summary analysis

**Without Debugger:**
- Failures silently discarded
- Same mistakes repeated (we saw `[z-a]` error twice)
- No learning across generations

#### Plateau Breaker Agent: 1 Invocation

**Trigger**: Generation 4, after 3 generations with 0% improvement

**Diagnosis**:
```
Evolution stuck in local optimum
Current pattern: 'Hope|Empire|Jedi|Phantom|Clones|Sith'
Current fitness: 964
```

**Proposed Interventions**:
1. `[algorithm_swap]` Try completely different matching strategy
2. `[paradigm_shift]` Use lookahead/lookbehind instead of alternation
3. `[structural]` Identify minimal distinguishing features

**Recommended Intervention**: Try pattern `ope|edi|nac|ith|ack|lon`

**Result**: ACCEPTED! Fitness improved 964 → 977

**What the Plateau Breaker provides:**
1. **Automatic stall detection** - configurable threshold (default <2% for 3 gens)
2. **Structured diagnosis** - explains WHY evolution is stuck
3. **Radical interventions** - not tweaks, but paradigm shifts
4. **Risk assessment** - ranked by expected impact and risk level

**Without Plateau Breaker:**
- Would continue trying incremental mutations indefinitely
- Human must manually diagnose stall
- No systematic exploration of alternatives

## Fitness Trajectory

```
Generation 1: 964          (baseline holds)
Generation 2: 964          (no improvement)
Generation 3: 964          (no improvement)
Generation 4: 964 → 977    [PLATEAU DETECTED → BREAKTHROUGH]
              ↑
              Plateau Breaker intervention
```

## Quantified Value

| Metric | Without Agents | With Agents | Impact |
|--------|----------------|-------------|--------|
| Failed mutations | Silently discarded | Diagnosed & categorized | Learning enabled |
| Plateau detection | Manual (gen 10+?) | Automatic (gen 4) | 6+ generations saved |
| Intervention | Human brainstorming | Structured proposals | Faster breakthrough |
| Final solution | Stuck at 964 | Reached 977 | +13 fitness points |

## Files

```
showcase/regex_golf/
├── README.md              # This file
├── problem.py             # Problem definitions (Star Wars vs Trek + others)
├── evaluator.py           # Fitness evaluation (correctness + length)
├── gen0_baseline.py       # Starting solution
├── run_evolution.py       # Evolution runner demonstrating agents
└── evolution_log.json     # Detailed run log
```

## Running the Showcase

```bash
cd sdk
source .venv/bin/activate  # or your venv
python showcase/regex_golf/run_evolution.py
```

## Key Takeaways

1. **Debugger Agent catches 33% of mutations** before they waste evaluation cycles
2. **Plateau Breaker detects stalls automatically** at generation 4
3. **Radical intervention breaks through** where incremental changes failed
4. **The agents don't replace the LLM** - they give it better information to make decisions

The prompts generated by these agents provide structured context that would be impossible to maintain manually across generations. This is the value of Phase 1.
