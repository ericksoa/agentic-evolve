# Roadmap Implementation Summary

## Quick Overview

**Goal**: Add intelligent strategy selection and hyperparameter evolution to evolve-sdk

**Why**: Santa 2025 analysis showed sparrow algorithm with default params failed, but with optimized params would likely have won 3rd place. SDK evolves code but not hyperparameters.

**Timeline**: 3 weeks (15 working days)

---

## The Problem We're Solving

```
Current SDK:
- Evolves algorithm CODE ✅
- Does NOT evolve algorithm HYPERPARAMETERS ❌

Example:
- Santa Gen100: Sparrow with defaults → 13% worse
- Likely winner: Sparrow with optimized params → 3rd place

Gap: No way to optimize the 8 parameters in SparrowConfig:
  exploration_iters, compression_iters, base_penalty,
  penalty_growth, penalty_decay, max_penalty, etc.
```

---

## The Solution: 4 Phases

### Phase 1: Strategy Selection (Week 1, 7 days)
**What**: SDK automatically detects if code has parameters and recommends strategy

**How**:
- Analyze problem type (new algorithm vs tune existing)
- Detect parameters in code using Claude
- Query memory for what worked on similar problems
- Ask user for confirmation

**Files**: 7 new, 3 modified, 5 test files

**Key Test**: Detect sparrow has 8 parameters → recommend hyperparameter evolution

---

### Phase 2: Plateau Detection (Week 2, 4 days)
**What**: Automatically detect when stuck and switch strategies

**How**:
- Monitor fitness progression
- Check mutation diversity
- Use adversary to validate plateau
- Ask user before switching

**Files**: 2 new, 3 modified, 4 test files

**Key Test**: Detect Gen92-100 plateau → suggest trying hyperparameter evolution

---

### Phase 3: Hyperparameter Evolution (Week 3, 7 days)
**What**: Optimize algorithm parameters using CMA-ES/Bayesian optimization

**How**:
- Detect parameters in code
- Run black-box optimization (CMA-ES)
- Trust system validates improvements
- Apply optimized params back to code

**Files**: 5 new, 4 modified, 6 test files

**Key Test**: Optimize sparrow's 8 parameters → >10% improvement over defaults

---

### Phase 4: Integration (Week 4, 3 days)
**What**: Seamless end-to-end experience

**How**:
- Unified orchestrator
- Enhanced UI with progress
- Error recovery
- Documentation

**Files**: 8 new, 2 modified, 4 test files

**Key Test**: Run complete santa optimization from scratch → validate against Gen100-120

---

## File Statistics

| Phase | New Files | Modified | Tests | Total LOC |
|-------|-----------|----------|-------|-----------|
| Phase 1 | 7 | 3 | 5 | ~1500 |
| Phase 2 | 2 | 3 | 4 | ~1000 |
| Phase 3 | 5 | 4 | 6 | ~2000 |
| Phase 4 | 8 | 2 | 4 | ~1500 |
| **Total** | **22** | **12** | **19** | **~6000** |

Plus ~1000 lines of documentation and examples.

---

## Validation Strategy

Each phase has concrete, testable success criteria:

### Phase 1: Can detect parameters and recommend strategies
```python
# Test passes if:
assert context.has_parameters == True  # Detects sparrow has 8 params
assert decision.strategy == "hyperparameter_evolution"  # Recommends right strategy
```

### Phase 2: Can detect plateau and switch
```python
# Test passes if:
assert plateau.is_plateaued == True  # Detects 20 gens without improvement
assert switch.to_strategy == "hyperparameter_evolution"  # Switches correctly
```

### Phase 3: Can optimize parameters
```python
# Test passes if:
assert abs(result.best_parameters['lr'] - 2.0) < 0.1  # Finds optimal value
assert result.improvement > 0.1  # >10% improvement
```

### Phase 4: Works end-to-end
```python
# Test passes if:
assert result.champion.fitness < 12.0  # Beats Gen100 sparrow (12.33)
assert overhead < 0.10  # <10% performance overhead
```

---

## Dependencies

### Python Packages (add to pyproject.toml)
```toml
[project.dependencies]
# Phase 1
prompt-toolkit = "^3.0.0"

# Phase 3
cma = "^3.3.0"
scikit-optimize = "^0.9.0"
numpy = "^1.24.0"
scipy = "^1.10.0"
```

### SDK Systems (all existing)
- Memory system ✅
- Adversary agent ✅
- Trust system ✅
- Agent SDK ✅

---

## Implementation Order

See `IMPLEMENTATION_ORDER.md` for detailed day-by-day breakdown.

**TL;DR per phase:**
1. Create types and data structures
2. Implement core logic
3. Integrate with existing systems (memory, adversary, trust)
4. Write tests proving it works
5. Integrate into main evolution loop
6. Validate with real-world test

---

## Success Metrics

### Phase 1 Success
- ✅ Correctly classifies 4 problem types
- ✅ Detects parameters in various code patterns
- ✅ Memory enhances recommendations
- ✅ User receives clear explanations

### Phase 2 Success
- ✅ Plateau detected after patience period
- ✅ Adversary validates claims
- ✅ Already-tried strategies filtered
- ✅ User confirms before switching

### Phase 3 Success
- ✅ Parameters detected in complex code
- ✅ CMA-ES optimizes test functions
- ✅ Trust system catches exploits
- ✅ Parameters correctly applied to code

### Phase 4 Success
- ✅ End-to-end flow seamless
- ✅ Graceful error recovery
- ✅ Performance overhead <10%
- ✅ Santa challenge validation passes

---

## Risk Mitigation

### Risk: Parameter detection fails
**Mitigation**: Fall back to code evolution, warn user

### Risk: Optimization takes too long
**Mitigation**: User confirms budget upfront, can cancel anytime

### Risk: Trust system rejects improvements
**Mitigation**: Show user why rejected, allow override with confirmation

### Risk: Memory corruption
**Mitigation**: Continue without history, warn user

### Risk: Phase 3 delays whole project
**Mitigation**: Phases 1-2 are independently useful, can ship without Phase 3

---

## Real-World Validation Plan

Each phase validated on Santa 2025:

**Phase 1**: Detect sparrow has parameters ✅
**Phase 2**: Detect Gen92-100 plateau ✅
**Phase 3**: Optimize sparrow params, beat Gen100 score ✅
**Phase 4**: Complete run from scratch, match what Gen100-120 should have been ✅

Final test:
```bash
cd showcase/santa-2025-packing
python -m evolve_sdk "optimize sparrow algorithm for santa challenge"
# Should find score < 80 (vs Gen100's 12.33 failure)
```

---

## Documentation Deliverables

### User-Facing Docs
- `docs/QUICKSTART.md` - Getting started
- `docs/STRATEGIES.md` - Strategy selection guide
- `docs/HYPERPARAMETERS.md` - Hyperparameter evolution guide
- `docs/TROUBLESHOOTING.md` - Common issues

### Examples
- `examples/simple_optimization.py` - Basic usage
- `examples/santa_hyperparameter.py` - Real-world example
- `examples/advanced_strategies.py` - All features

### README Updates
- New capabilities section
- Updated usage examples
- Architecture diagram updates

---

## Post-Implementation: Future Roadmap

After Phase 4 completes, these become possible:

### Cloud Execution (separate roadmap)
- AWS Batch, Modal, Lightning.ai integrations
- User chooses local vs cloud
- Cost estimation before execution

### Population-Based Evolution
- Maintain multiple solutions
- Crossover between solutions
- Ensemble methods

### Transfer Learning
- Learn parameter ranges across problems
- Strategy patterns that generalize
- Cross-domain knowledge

---

## Quick Start After Context Clear

1. Read `roadmap/README.md` - Overview
2. Read `roadmap/PHASE1_STRATEGY_SELECTION.md` - Current phase
3. Read `roadmap/IMPLEMENTATION_ORDER.md` - Detailed steps
4. Check file checklist to see progress
5. Run tests to verify what's working
6. Continue from next incomplete file

---

## Key Insights from Santa RCA

1. **Meta-algorithm hyperparameters matter MORE than algorithm code**
   - Sparrow with defaults: Failed (13% worse)
   - Sparrow with optimal params: Likely won 3rd place
   
2. **SDK only evolved code, not hyperparameters**
   - This was the critical missing capability
   
3. **Each phase is independently valuable**
   - Phase 1 alone: Better strategy selection
   - Phases 1+2: Auto-switching when stuck
   - Phases 1+2+3: Full hyperparameter evolution
   
4. **All SDK systems must work together**
   - Memory: Learn from past
   - Adversary: Validate decisions
   - Trust: Prevent exploits
   - UI: Clear communication

---

## Contact / Questions

After reading all roadmap docs, if you have questions:
1. Check `roadmap/PHASE[N]_*.md` for detailed phase info
2. Check `roadmap/IMPLEMENTATION_ORDER.md` for specific file details
3. Check existing SDK code for integration examples
