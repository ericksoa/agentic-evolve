# Evolve SDK Roadmap

This roadmap outlines the phased development of intelligent strategy selection and hyperparameter evolution for the Evolve SDK.

## Background

Analysis of the Santa 2025 Kaggle challenge revealed a critical gap: the SDK evolved **algorithm code** but not **algorithm hyperparameters**. The sparrow algorithm with default parameters performed poorly (13% worse), but with optimized parameters it likely would have won 3rd place.

This roadmap addresses that gap by adding:
1. Intelligent problem analysis and strategy selection
2. Automatic plateau detection and strategy switching
3. Hyperparameter evolution using black-box optimization
4. Seamless integration of all capabilities

## Roadmap Phases

### ✅ Phase 0: Current State (Completed)
- Code evolution via mutation agents
- Memory system for learning
- Adversary agent for validation
- Trust system for preventing exploits
- Basic evaluation framework

### 🔴 Phase 1: Strategy Selection (1 week)
**File**: [`PHASE1_STRATEGY_SELECTION.md`](./PHASE1_STRATEGY_SELECTION.md)

Enable automatic problem analysis and strategy recommendation:
- Detect problem type (new algorithm vs tune existing)
- Identify tunable parameters in code
- Query memory for similar past successes
- Ask user for confirmation with clear explanations

**Key Deliverables**:
- `evolve_sdk/strategy/problem_analyzer.py`
- `evolve_sdk/strategy/strategy_selector.py`
- `evolve_sdk/ui/user_interface.py`

**Validation**: 4 tests proving classification, parameter detection, memory learning, user confirmation

---

### 🔴 Phase 2: Plateau Detection (4 days)
**File**: [`PHASE2_PLATEAU_DETECTION.md`](./PHASE2_PLATEAU_DETECTION.md)

Automatically detect when stuck and switch strategies:
- Monitor fitness progression
- Analyze mutation diversity
- Use adversary to validate plateau claims
- Auto-switch with user permission

**Key Deliverables**:
- `evolve_sdk/strategy/plateau_detector.py`
- `evolve_sdk/strategy/auto_switcher.py`

**Validation**: 5 tests proving plateau detection, adversary validation, strategy filtering, user confirmation

---

### 🔴 Phase 3: Hyperparameter Evolution (1 week)
**File**: [`PHASE3_HYPERPARAMETER_EVOLUTION.md`](./PHASE3_HYPERPARAMETER_EVOLUTION.md)

Evolve algorithm hyperparameters using black-box optimization:
- Detect tunable parameters in code
- Optimize using CMA-ES, Bayesian, or random search
- Validate improvements with trust system
- Apply optimized parameters back to code

**Key Deliverables**:
- `evolve_sdk/hyperparameter/detector.py`
- `evolve_sdk/hyperparameter/evolver.py`
- `evolve_sdk/hyperparameter/search_methods.py`

**Validation**: 6 tests proving parameter detection, optimization, trust validation, code application

---

### 🔴 Phase 4: Integration & UX (3 days)
**File**: [`PHASE4_INTEGRATION.md`](./PHASE4_INTEGRATION.md)

Seamless end-to-end experience:
- Unified orchestration of all strategies
- Enhanced user interface
- Error recovery and graceful degradation
- Comprehensive documentation

**Key Deliverables**:
- `evolve_sdk/orchestrator.py`
- `evolve_sdk/ui/enhanced_ui.py`
- Documentation and examples

**Validation**: 6 tests proving end-to-end flow, error recovery, resume, performance, real-world validation

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1 | 1 week | Week 1 | Week 1 |
| Phase 2 | 4 days | Week 2 | Week 2 |
| Phase 3 | 1 week | Week 3 | Week 3 |
| Phase 4 | 3 days | Week 4 | Week 4 |
| **Total** | **~3 weeks** | - | - |

## Success Metrics

### Phase 1 Success
- ✅ 4 problem types correctly classified
- ✅ Parameters detected in various code patterns
- ✅ Memory enhances recommendations (>3 samples)
- ✅ User receives clear strategy explanations

### Phase 2 Success
- ✅ Plateau detected after patience period
- ✅ Adversary validates claims
- ✅ Already-tried strategies filtered out
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
- ✅ Validated on santa-2025-packing

## Real-World Validation

Each phase will be validated on the Santa 2025 packing challenge:

**Phase 1**: Detect that sparrow algorithm has tunable parameters, recommend hyperparameter evolution

**Phase 2**: Detect when code evolution plateaus (Gen92-100), suggest trying hyperparameter evolution

**Phase 3**: Optimize sparrow parameters, achieve >10% improvement over Gen100 default params

**Phase 4**: Complete orchestrated run from scratch, reproduce what Gen100-120 should have been

## Future Roadmap (Post-Phase 4)

### Cloud Execution
**File**: `roadmap/FUTURE_CLOUD_EXECUTION.md` (to be created)

Optional cloud backends for heavy compute:
- AWS Batch, Modal, Lightning.ai integrations
- User chooses local vs cloud per run
- Cost estimation before execution
- Auto-scaling based on budget

### Population-Based Evolution
**File**: `roadmap/FUTURE_POPULATION.md` (to be created)

Maintain multiple solutions simultaneously:
- Crossover between different solutions
- Diversity preservation
- Ensemble methods

### Transfer Learning
**File**: `roadmap/FUTURE_TRANSFER_LEARNING.md` (to be created)

Learn across problem domains:
- Parameter ranges that work across problems
- Strategy selection patterns
- Mutation patterns that generalize

## Dependencies

### External Libraries
```toml
# Phase 1
prompt-toolkit = "^3.0.0"  # Interactive prompts

# Phase 3
cma = "^3.3.0"              # CMA-ES optimization
scikit-optimize = "^0.9.0"  # Bayesian optimization
numpy = "^1.24.0"
scipy = "^1.10.0"
```

### Internal Components
- Memory system (existing)
- Adversary agent (existing)
- Trust system (existing)
- Agent SDK (existing)

## Testing Strategy

Each phase has:
1. **Unit tests** - Individual components
2. **Integration tests** - Component interactions
3. **End-to-end tests** - Full workflows
4. **Validation test** - Proves phase works

All tests must pass before proceeding to next phase.

## Rollout Strategy

1. **Week 1**: Implement Phase 1
2. **Week 2**: Implement Phase 2
3. **Week 3**: Implement Phase 3
4. **Week 4**: Implement Phase 4
5. **Week 5**: Beta test on all showcases
6. **Week 6**: Stable release

Each phase is independently useful:
- Phase 1 alone: Better strategy selection
- Phases 1+2: Auto-switching when stuck
- Phases 1+2+3: Full hyperparameter evolution
- Phases 1+2+3+4: Production-ready experience

## Questions & Decisions

**Q**: Should cloud execution be in Phase 4 or separate?
**A**: Separate roadmap. Phase 4 focuses on local execution UX. Cloud is a major undertaking that should be planned carefully with user choice at the center.

**Q**: Can phases be worked on in parallel?
**A**: No. Each phase builds on previous. Must be sequential.

**Q**: What if Phase 3 takes longer than expected?
**A**: Phase 3 can ship without Phase 4. Integration is polish, not requirement.

## Notes

- Each phase is independently valuable
- Tests prove each phase works before moving forward
- Real-world validation on santa challenge critical
- Focus on library features, not infrastructure (cloud separate)
- User choice and clarity always prioritized
