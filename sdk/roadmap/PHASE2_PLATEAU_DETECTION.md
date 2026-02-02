# Phase 2: Plateau Detection & Auto Strategy Switching

**Status**: 🔴 Not Started
**Priority**: P0
**Estimated Duration**: 4 days
**Dependencies**: Phase 1 (Strategy Selection)

## Overview

Automatically detect when evolution is stuck in a plateau and intelligently switch strategies. The system will:
- Monitor fitness progression across generations
- Detect when improvement has stalled
- Use adversary agent to validate plateau (not evaluator bug)
- Ask user permission before switching strategies
- Avoid suggesting strategies already tried
- Learn which strategy switches work best

## Goals

- ✅ Automatic plateau detection based on fitness history
- ✅ Mutation diversity analysis (detect loss of exploration)
- ✅ Adversary validation of plateau claims
- ✅ Smart strategy switching with user confirmation
- ✅ Memory of tried strategies to avoid repetition
- ✅ Learning from successful/failed strategy switches

## Architecture Components

### 1. Plateau Detector
**File**: `evolve_sdk/strategy/plateau_detector.py`

Monitors evolution progress and detects plateaus:
- Analyzes recent generation fitness from memory
- Checks mutation diversity (are mutations too similar?)
- Uses adversary to validate plateau is real (not evaluator bug)
- Returns plateau detection with confidence and suggested actions

**Key Methods**:
```python
def check_plateau(generation: int) -> PlateauDetection
def _check_mutation_diversity(recent_gens: List[MemoryFrame]) -> float
def _suggest_actions(improvement: float, diversity: float) -> List[str]
```

**Configuration**:
```python
patience: int = 20           # Generations without improvement
min_improvement: float = 0.01  # 1% minimum improvement threshold
min_diversity: float = 0.3     # Mutation diversity threshold
```

### 2. Auto Strategy Switcher
**File**: `evolve_sdk/strategy/auto_switcher.py`

Handles automatic strategy switching:
- Checks for plateau using detector
- Filters out already-tried strategies
- Asks user permission to switch
- Records switch in memory for learning

**Key Methods**:
```python
def check_and_switch(generation: int, current_strategy: str) -> Optional[StrategySwitch]
def _filter_tried_strategies(actions: List[str]) -> List[str]
def _propose_switch(current_strategy, plateau, available_actions) -> StrategySwitch
```

### 3. Supporting Types
**File**: `evolve_sdk/strategy/types.py` (extend)

New data structures:
```python
@dataclass
class PlateauDetection:
    is_plateaued: bool
    generations_stuck: int
    suggested_actions: List[str]
    confidence: float
    warning: Optional[str] = None

@dataclass
class StrategySwitch:
    from_strategy: str
    to_strategy: str
    reason: str
    generation: int
```

## Integration Points

### With Adversary Agent
**What**: Validate that plateau is real, not evaluator issues

**Where**: `evolve_sdk/agents/adversary.py`

**New Method**:
```python
def validate_plateau(recent_generations: List[MemoryFrame],
                    claimed_plateau: bool) -> ValidationResult:
    """
    Check if plateau is genuine or caused by:
    - Evaluator bugs/inconsistency
    - Test suite changes
    - Random noise masking improvements
    """
```

### With Memory System
**What**: Query generation history, store strategy switches

**Where**: `evolve_sdk/memory/schemas.py`

**New Frame Type**:
```python
{
    "frame_type": "strategy_switch",
    "metadata": {
        "generation": 45,
        "from_strategy": "code_evolution",
        "to_strategy": "hyperparameter_evolution",
        "reason": "Plateau detected after 20 generations",
        "plateau_confidence": 0.9
    }
}
```

### With Main Evolution Loop
**What**: Check plateau and switch at each generation

**Where**: `evolve_sdk/evolution.py`

**Changes**:
```python
for generation in range(max_generations):
    # PHASE 2: Check for plateau and auto-switch
    switch = self.auto_switcher.check_and_switch(generation, current_strategy)

    if switch:
        logger.info(f"Switching from {switch.from_strategy} to {switch.to_strategy}")
        current_strategy = switch.to_strategy
        self.memory.store_strategy_switch(switch)

    # Continue with current strategy
    ...
```

## Files to Create

```
sdk/evolve_sdk/strategy/
├── plateau_detector.py       # Plateau detection logic
└── auto_switcher.py          # Automatic strategy switching

sdk/tests/strategy/
├── test_plateau_detector.py
└── test_auto_switcher.py
```

## Files to Modify

- `evolve_sdk/strategy/types.py` - Add PlateauDetection, StrategySwitch types
- `evolve_sdk/agents/adversary.py` - Add validate_plateau() method
- `evolve_sdk/memory/schemas.py` - Add strategy_switch frame type
- `evolve_sdk/memory/queries.py` - Add query for tried strategies
- `evolve_sdk/evolution.py` - Integrate plateau checking in main loop

## Validation Tests

### Test 1: Plateau Detection Triggers
```python
def test_plateau_detection_triggers():
    """Plateau detected after N generations without improvement."""
    memory = MemoryManager()
    adversary = MockAdversaryAgent()
    detector = PlateauDetector(memory, adversary)

    # Simulate 20 generations with no improvement
    for gen in range(20):
        memory.store(MemoryFrame(
            frame_type="generation",
            metadata={"generation": gen, "fitness": 1.5}  # Same every time
        ))

    plateau = detector.check_plateau(generation=20)

    assert plateau.is_plateaued == True
    assert plateau.generations_stuck == 20
    assert "hyperparameter_evolution" in plateau.suggested_actions
```

### Test 2: Adversary Validation
```python
def test_adversary_validates_plateau():
    """Adversary validates plateau before triggering switch."""
    adversary = MockAdversaryAgent()
    adversary.set_validation_result(is_valid=False)  # Claims evaluator bug

    detector = PlateauDetector(memory, adversary)

    # Simulate plateau
    for gen in range(20):
        memory.store(MemoryFrame(
            frame_type="generation",
            metadata={"generation": gen, "fitness": 1.5}
        ))

    plateau = detector.check_plateau(generation=20)

    # Should NOT be plateaued due to adversary rejection
    assert plateau.is_plateaued == False
    assert "evaluator issues" in plateau.warning
```

### Test 3: Filter Tried Strategies
```python
def test_auto_switcher_filters_tried_strategies():
    """Don't suggest strategies already tried."""
    memory = MemoryManager()

    # Record that we already tried hyperparameter evolution
    memory.store(MemoryFrame(
        frame_type="generation",
        metadata={"generation": 50, "strategy": "hyperparameter_evolution"}
    ))

    switcher = AutoStrategySwitcher(detector, selector, ui, memory)

    available = switcher._filter_tried_strategies([
        "hyperparameter_evolution",
        "alternative_paradigm"
    ])

    assert "hyperparameter_evolution" not in available
    assert "alternative_paradigm" in available
```

### Test 4: User Confirmation Required
```python
def test_user_confirms_strategy_switch():
    """User must confirm before switching strategies."""
    ui = MockUserInterface(choice="hyperparameter_evolution")
    switcher = AutoStrategySwitcher(detector, selector, ui, memory)

    # Trigger plateau
    plateau = PlateauDetection(
        is_plateaued=True,
        generations_stuck=25,
        suggested_actions=["hyperparameter_evolution"],
        confidence=0.9
    )

    switch = switcher._propose_switch("code_evolution", plateau, ["hyperparameter_evolution"])

    assert ui.was_asked == True
    assert switch.to_strategy == "hyperparameter_evolution"
```

### Test 5: Diversity Check
```python
def test_mutation_diversity_calculation():
    """Mutation diversity correctly computed."""
    # Create mutations with varying similarity
    mutations = [
        create_mutation("def f(x): return x * 2"),
        create_mutation("def f(x): return x * 2 + 1"),  # Similar
        create_mutation("def g(x): return x ** 3"),      # Different
    ]

    for m in mutations:
        memory.store(m)

    detector = PlateauDetector(memory, adversary)
    diversity = detector._check_mutation_diversity(recent_gens=[...])

    assert 0.0 <= diversity <= 1.0
    # With mix of similar and different, expect medium diversity
    assert 0.3 < diversity < 0.7
```

## Success Criteria

- [x] Plateau detection triggers after configurable patience period
- [x] Mutation diversity correctly identifies loss of exploration
- [x] Adversary validates plateau claims before switching
- [x] User is asked and confirms before strategy switches
- [x] Already-tried strategies are not suggested
- [x] Strategy switches are recorded in memory
- [x] All tests pass
- [x] No regressions in existing functionality

## Implementation Checklist

### Step 1: Extend Types (Day 1 Morning)
- [ ] Add `PlateauDetection` and `StrategySwitch` to types.py
- [ ] Add docstrings and examples

### Step 2: Plateau Detector (Day 1-2)
- [ ] Create `plateau_detector.py`
- [ ] Implement `check_plateau()` with fitness analysis
- [ ] Implement `_check_mutation_diversity()` using edit distance
- [ ] Implement `_suggest_actions()` based on improvement/diversity
- [ ] Write unit tests for detector

### Step 3: Adversary Integration (Day 2)
- [ ] Add `validate_plateau()` to adversary agent
- [ ] Implement plateau validation logic
- [ ] Test adversary can catch evaluator issues

### Step 4: Auto Switcher (Day 3)
- [ ] Create `auto_switcher.py`
- [ ] Implement `check_and_switch()` orchestration
- [ ] Implement `_filter_tried_strategies()` using memory
- [ ] Implement `_propose_switch()` with user confirmation
- [ ] Write unit tests for switcher

### Step 5: Memory Integration (Day 3)
- [ ] Add `strategy_switch` frame type to schemas
- [ ] Add query method for tried strategies
- [ ] Test memory records switches correctly

### Step 6: Evolution Loop Integration (Day 4)
- [ ] Add plateau checker and switcher to evolution loop
- [ ] Integrate at start of each generation
- [ ] Add configuration options
- [ ] Write integration tests

### Step 7: Testing & Polish (Day 4)
- [ ] End-to-end testing with real evolution runs
- [ ] Test edge cases (all strategies tried, user says stop, etc.)
- [ ] Update documentation
- [ ] Final bug fixes

## Non-Goals (Out of Scope)

- ❌ Implementing new strategies themselves (Phase 3, 4)
- ❌ ML-based plateau prediction
- ❌ Automatic parameter tuning for plateau detector
- ❌ Multi-metric plateau detection (only fitness for now)

## Configuration Options

Add to `evolve_config.json`:
```json
{
  "plateau_detection": {
    "enabled": true,
    "patience": 20,
    "min_improvement": 0.01,
    "min_diversity": 0.3,
    "require_adversary_validation": true,
    "auto_switch": true,
    "ask_user_confirmation": true
  }
}
```

## Edge Cases to Handle

1. **All strategies exhausted**: Ask user what to do (continue, stop, custom)
2. **User declines switch**: Continue with current strategy
3. **Adversary unavailable**: Warn user, proceed without validation
4. **Memory corrupted**: Fall back to no history (treat as first run)
5. **Plateau immediately**: Don't switch if <10 generations

## Future Enhancements (Post-Phase 2)

- Multi-metric plateau detection (fitness + diversity + other signals)
- Adaptive patience (longer for complex problems)
- Plateau prediction before it happens
- Strategy effectiveness scoring over time
- Automatic parameter tuning for detector thresholds

## Dependencies

**SDK Components**:
- Phase 1 (Strategy Selection) - REQUIRED
- Memory system - REQUIRED
- Adversary agent - REQUIRED
- User interface - REQUIRED

**Python Packages**: None new

## Migration Notes

**Breaking Changes**: None

**Backward Compatibility**:
- Plateau detection is opt-in via config
- Default: enabled but asks user before switching
- Can disable entirely with `plateau_detection.enabled = false`

## Rollout Plan

1. **Day 1-3**: Implementation
2. **Day 4**: Testing and integration
3. **Day 5**: Beta test on santa-2025-packing showcase
4. **Day 6**: Iterate based on feedback
5. **Day 7**: Release to all showcases

## Notes

- Plateau detection should be conservative (better to miss plateau than false positive)
- Always ask user before switching - don't surprise them
- Clear messaging about why switching and what's changing
- Record every switch for post-mortem analysis
