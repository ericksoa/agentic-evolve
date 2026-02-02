# Start Phase 1: Strategy Selection

**Copy and paste this entire file as your prompt after context clear.**

---

## Context

I'm working on the **Evolve SDK** - a Python library that uses Claude to evolve algorithms through mutation and selection.

**Current location**: `/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/sdk`

**Problem**: The SDK currently evolves algorithm **code** but not algorithm **hyperparameters**. Root cause analysis of the Santa 2025 Kaggle challenge showed this was a critical gap - the sparrow algorithm with default parameters performed 13% worse than baseline, but with optimized parameters it likely would have won 3rd place.

**Solution**: 4-phase roadmap to add intelligent strategy selection and hyperparameter evolution.

---

## What I Need You To Do

I'm starting **Phase 1: Strategy Selection** (Week 1, 7 days).

**Goal**: Enable the SDK to automatically:
1. Analyze the user's problem and code
2. Detect if the code has tunable parameters (like `learning_rate=0.01`, `iterations=1000`)
3. Query memory for what worked on similar problems
4. Recommend the best strategy (code evolution vs hyperparameter evolution)
5. Ask the user for confirmation with clear explanations

---

## Your Task

Start implementing Phase 1 following the detailed roadmap.

**Step 1**: Read the roadmap documents to understand the plan:

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/sdk

# Read these in order:
cat roadmap/SUMMARY.md              # 5 min - Quick overview
cat roadmap/PHASE1_STRATEGY_SELECTION.md  # 15 min - Detailed Phase 1 spec
cat roadmap/IMPLEMENTATION_ORDER.md  # Reference - Day-by-day guide
```

**Step 2**: After reading, start Day 1, Step 1.1 from `IMPLEMENTATION_ORDER.md`:

```
Day 1, Step 1.1: Create Directory Structure
mkdir -p evolve_sdk/strategy
mkdir -p evolve_sdk/ui
mkdir -p tests/strategy
mkdir -p tests/ui
```

**Step 3**: Continue with Day 1, Step 1.2 - Create base types in `evolve_sdk/strategy/__init__.py` and `evolve_sdk/strategy/types.py`.

---

## Implementation Guidelines

### Code Style
- Follow existing SDK patterns (check `evolve_sdk/agents/` for examples)
- Use type hints everywhere
- Add comprehensive docstrings
- Keep functions focused and small (50-100 lines max)

### Testing
- Write tests as you go (TDD approach)
- Each test should be clear and self-contained
- Use pytest fixtures for common setup
- Test file naming: `test_<module_name>.py`

### Integration with Existing Systems
Phase 1 integrates with these **existing** SDK systems:
- **Memory system**: `evolve_sdk/memory/` (query for similar problems)
- **User interface**: Create new `evolve_sdk/ui/` module
- **Runner**: `evolve_sdk/runner.py` (minimal modifications)

### File Sizes (for planning)
- Types: ~100 lines
- Problem Analyzer: ~200 lines
- Strategy Selector: ~250 lines
- User Interface: ~150 lines
- Tests: ~100-200 lines each

### Validation Gates
Before moving to Phase 2, all Phase 1 tests must pass:
```bash
pytest tests/strategy/ -v
pytest tests/ui/ -v
pytest tests/integration/test_phase1_integration.py -v
```

---

## Key Files to Create (in order)

**Day 1** (Types & Setup):
1. `evolve_sdk/strategy/__init__.py`
2. `evolve_sdk/strategy/types.py`
3. `tests/strategy/test_types.py`

**Day 2** (Problem Analyzer):
4. `evolve_sdk/strategy/problem_analyzer.py`
5. `tests/strategy/test_problem_analyzer.py`

**Day 3** (User Interface):
6. `evolve_sdk/ui/__init__.py`
7. `evolve_sdk/ui/user_interface.py`
8. `tests/ui/test_user_interface.py`

**Day 4** (Strategy Selector):
9. `evolve_sdk/strategy/strategy_selector.py`
10. `tests/strategy/test_strategy_selector.py`

**Day 5** (Memory Integration):
- Modify: `evolve_sdk/memory/queries.py` (add ~100 lines)
- Modify: `evolve_sdk/memory/schemas.py` (add ~50 lines)

**Day 6** (Runner Integration):
- Modify: `evolve_sdk/runner.py` (add ~50 lines at start of run())
- Create: `tests/integration/test_phase1_integration.py`

**Day 7** (Testing & Docs):
- Run full test suite
- Create: `examples/phase1_demo.py`
- Update: `README.md`

---

## Important Notes

### DO:
- ✅ Read the roadmap files first to understand the full picture
- ✅ Follow the implementation order exactly (it's been validated)
- ✅ Write tests as you implement each component
- ✅ Ask me questions if anything is unclear
- ✅ Validate each step before moving to the next

### DON'T:
- ❌ Skip ahead to later phases (they depend on Phase 1)
- ❌ Modify existing SDK core systems (agents, trust, etc.) - Phase 1 doesn't touch these
- ❌ Create new files not listed in the roadmap
- ❌ Change the validation test criteria (they're specifically designed)

---

## Dependencies

**Already installed** (existing SDK):
- claude-agent-sdk
- Standard Python libraries

**To install for Phase 1**:
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/sdk
source .venv/bin/activate  # or create venv if needed
pip install prompt-toolkit  # For interactive user prompts
```

---

## Success Criteria (How you'll know Phase 1 is done)

Run these tests - all must pass:

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/sdk

# Phase 1 tests
pytest tests/strategy/test_types.py -v
pytest tests/strategy/test_problem_analyzer.py -v
pytest tests/ui/test_user_interface.py -v
pytest tests/strategy/test_strategy_selector.py -v
pytest tests/integration/test_phase1_integration.py -v

# No regressions in existing tests
pytest tests/ -v
```

**Manual validation**: Create a test script that:
1. Loads code with parameters (like Santa sparrow algorithm)
2. Runs problem analyzer
3. Shows it detects parameters
4. Shows it recommends hyperparameter evolution
5. Shows clear user messaging

---

## Example: What Phase 1 Will Enable

**Before Phase 1**:
```python
# User runs evolution
python -m evolve_sdk "optimize my algorithm"
# SDK always does code evolution, never considers hyperparameters
```

**After Phase 1**:
```python
# User runs evolution
python -m evolve_sdk "optimize my algorithm"

# SDK analyzes code, detects:
# - Has parameters: learning_rate=0.01, iterations=1000, threshold=0.5
# - Similar past problem: hyperparameter evolution gave 15% improvement
# - Recommendation: Try hyperparameter evolution (85% confidence)

# Asks user:
# "Found 3 tunable parameters. Should I:
#  1. ✅ Optimize parameters (Recommended)
#  2. Evolve code structure
#  3. Custom settings"
```

---

## Quick Start Command

After reading the roadmap files, start here:

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/sdk

# Create directories
mkdir -p evolve_sdk/strategy evolve_sdk/ui tests/strategy tests/ui

# Start with types
# Create evolve_sdk/strategy/__init__.py and types.py
# Following Day 1, Step 1.2 from IMPLEMENTATION_ORDER.md
```

---

## Questions?

If anything is unclear:
1. First check `roadmap/PHASE1_STRATEGY_SELECTION.md` for details
2. Check `roadmap/IMPLEMENTATION_ORDER.md` for step-by-step guide
3. Check existing SDK code in `evolve_sdk/agents/` for patterns
4. Ask me to clarify specific aspects

---

## Ready to Start?

Confirm you've read:
- ✅ roadmap/SUMMARY.md (understand the why)
- ✅ roadmap/PHASE1_STRATEGY_SELECTION.md (understand Phase 1 spec)
- ✅ This START_PHASE1.md file (understand what to do)

Then say: "Ready to start Phase 1. Beginning with Day 1, Step 1.1 - creating directory structure."

I'll guide you through each step, making sure we stay on track with the roadmap.

---

## Checkpoint: After Reading

Tell me:
1. Do you understand **why** we're doing this? (Santa RCA - hyperparameter gap)
2. Do you understand **what** Phase 1 does? (Strategy selection with parameter detection)
3. Are you ready to start Day 1, Step 1.1?

Let's build this methodically, one step at a time. Phase 1 is ~1,500 LOC across 7 days. We'll validate as we go.
