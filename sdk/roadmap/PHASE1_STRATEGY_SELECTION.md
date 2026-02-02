# Phase 1: Intelligent Strategy Selection

**Status**: 🔴 Not Started
**Priority**: P0
**Estimated Duration**: 1 week
**Dependencies**: None

## Overview

Enable the SDK to automatically analyze problems and select the appropriate optimization strategy. The system will:
- Detect problem type (new algorithm, tune existing, stuck evolution)
- Identify tunable parameters in code
- Query memory for similar past successful runs
- Ask user for confirmation with clear explanations
- Learn from decisions to improve future recommendations

## Goals

- ✅ Automatic problem type classification
- ✅ Parameter detection in existing code
- ✅ Memory-enhanced strategy recommendations
- ✅ Clear user communication about strategy choices
- ✅ Learning from past strategy successes/failures

## Architecture Components

### 1. Problem Analyzer
**File**: `evolve_sdk/strategy/problem_analyzer.py`

Analyzes user prompts and existing solutions to determine:
- Problem type: `new_algorithm` | `tune_existing` | `stuck_evolution` | `hybrid`
- Has tunable parameters: `bool`
- Complexity estimate: `simple` | `medium` | `complex`
- Similar past runs from memory

**Key Methods**:
```python
def analyze(user_prompt: str, current_solution: Optional[Solution]) -> ProblemContext
def _detect_parameters(solution: Solution) -> bool
def _classify_problem_type(prompt: str, is_continuation: bool) -> str
```

### 2. Strategy Selector
**File**: `evolve_sdk/strategy/strategy_selector.py`

Recommends strategies and gets user confirmation:
- Generates ranked strategy recommendations
- Enhances with historical success data from memory
- Presents options to user with clear explanations
- Records decision in memory for future learning

**Key Methods**:
```python
def select(context: ProblemContext) -> StrategyDecision
def _recommend_strategies(context: ProblemContext) -> List[StrategyRecommendation]
def _enhance_with_memory(recommendations, context) -> List[StrategyRecommendation]
def _ask_user(recommendations, context) -> StrategyDecision
```

### 3. Supporting Types
**File**: `evolve_sdk/strategy/types.py`

Data structures:
```python
@dataclass
class ProblemContext:
    problem_type: str
    has_parameters: bool
    complexity: str
    similar_past_runs: List[MemoryFrame]
    is_continuation: bool

@dataclass
class StrategyRecommendation:
    strategy: str
    confidence: float
    reason: str

@dataclass
class StrategyDecision:
    strategy: str
    auto_selected: bool
    user_confirmed: bool
```

## Integration Points

### With Memory System
**What**: Query for similar past problems and their strategy outcomes

**Where**: `evolve_sdk/memory/queries.py`

**New Methods Needed**:
```python
def query_similar_problems(prompt: str) -> List[MemoryFrame]
def query_strategy_success_rate(strategy: str, keywords: List[str]) -> float
```

### With User Interface
**What**: Ask user to confirm strategy selection

**Where**: `evolve_sdk/ui/user_interface.py` (new file)

**Methods Needed**:
```python
def ask_user_choice(message: str, options: List[dict], default: str) -> str
def show_info(message: str)
def show_warning(message: str)
```

### With Main Evolution Loop
**What**: Call analyzer and selector before starting evolution

**Where**: `evolve_sdk/evolution.py` (main loop file)

**Changes**:
```python
# Add at start of run()
context = self.analyzer.analyze(user_prompt, current_solution)
decision = self.selector.select(context)
current_strategy = decision.strategy
```

## Files to Create

```
sdk/evolve_sdk/strategy/
├── __init__.py
├── problem_analyzer.py      # Problem analysis and classification
├── strategy_selector.py      # Strategy recommendation and selection
└── types.py                  # Data structures

sdk/evolve_sdk/ui/
├── __init__.py
└── user_interface.py         # User interaction methods

sdk/tests/strategy/
├── __init__.py
├── test_problem_analyzer.py
├── test_strategy_selector.py
└── test_integration.py
```

## Files to Modify

- `evolve_sdk/memory/queries.py` - Add similarity search methods
- `evolve_sdk/memory/schemas.py` - Add strategy_decision frame type
- `evolve_sdk/evolution.py` - Integrate analyzer/selector at start

## Validation Tests

### Test 1: New Algorithm Classification
```python
def test_strategy_selection_new_algorithm():
    """New algorithm request triggers code evolution."""
    context = analyzer.analyze("create faster sorting algorithm", None)

    assert context.problem_type == "new_algorithm"
    assert not context.is_continuation

    decision = selector.select(context)
    assert decision.strategy == "code_evolution"
```

### Test 2: Parameter Detection
```python
def test_strategy_selection_with_parameters():
    """Solution with parameters triggers hyperparameter evolution."""
    solution = Solution(code='''
class MySolver:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
    ''')

    context = analyzer.analyze("optimize this solver", solution)

    assert context.has_parameters == True
    assert context.problem_type == "tune_existing"

    decision = selector.select(context)
    assert decision.strategy == "hyperparameter_evolution"
```

### Test 3: Memory Learning
```python
def test_strategy_selection_learns_from_memory():
    """Memory influences strategy selection."""
    # Simulate past successful run with hyperparameter evolution
    memory.store(MemoryFrame(
        frame_type="generation",
        metadata={
            "strategy": "hyperparameter_evolution",
            "improvement": 0.15,
            "problem_keywords": ["sorting", "algorithm"]
        }
    ))

    # New similar problem
    context = analyzer.analyze("create faster sorting algorithm")
    recommendations = selector._recommend_strategies(context)
    enhanced = selector._enhance_with_memory(recommendations, context)

    # Should boost hyperparameter_evolution confidence
    hyper_rec = [r for r in enhanced if r.strategy == "hyperparameter_evolution"][0]
    assert "Historical success" in hyper_rec.reason
```

### Test 4: User Confirmation
```python
def test_user_is_asked_for_confirmation():
    """User is presented with recommendations and asked to choose."""
    ui = MockUserInterface(auto_select_first=True)
    selector = StrategySelector(memory, ui)

    context = ProblemContext(problem_type="new_algorithm", ...)
    decision = selector.select(context)

    # Verify user was asked
    assert ui.was_asked == True
    assert len(ui.options_shown) >= 1
    assert decision.user_confirmed == True
```

## Success Criteria

- [x] Problem analyzer classifies 4 problem types correctly
- [x] Parameter detection identifies tunable params in various code patterns
- [x] Strategy selector generates appropriate recommendations for each type
- [x] Memory enhances recommendations based on past success (>3 samples required)
- [x] User interface presents clear, understandable options
- [x] All tests pass
- [x] No regressions in existing functionality

## Implementation Checklist

### Step 1: Create Types (Day 1)
- [ ] Create `evolve_sdk/strategy/types.py`
- [ ] Define `ProblemContext`, `StrategyRecommendation`, `StrategyDecision`
- [ ] Add docstrings and type hints

### Step 2: Problem Analyzer (Day 2-3)
- [ ] Create `evolve_sdk/strategy/problem_analyzer.py`
- [ ] Implement `analyze()` method
- [ ] Implement `_detect_parameters()` using Claude
- [ ] Implement `_classify_problem_type()` with keyword matching
- [ ] Write unit tests

### Step 3: Strategy Selector (Day 3-4)
- [ ] Create `evolve_sdk/strategy/strategy_selector.py`
- [ ] Implement `_recommend_strategies()` with confidence scores
- [ ] Implement `_enhance_with_memory()` with success rate boosting
- [ ] Implement `_ask_user()` with formatted messages
- [ ] Write unit tests

### Step 4: User Interface (Day 4)
- [ ] Create `evolve_sdk/ui/user_interface.py`
- [ ] Implement `ask_user_choice()` with interactive prompts
- [ ] Implement info/warning message methods
- [ ] Create mock UI for testing

### Step 5: Memory Integration (Day 5)
- [ ] Add `query_similar_problems()` to memory queries
- [ ] Add `strategy_decision` frame type to schemas
- [ ] Test memory integration

### Step 6: Integration (Day 6)
- [ ] Update main evolution loop to use analyzer/selector
- [ ] Add configuration options for auto-selection
- [ ] Write integration tests
- [ ] Test end-to-end flow

### Step 7: Documentation & Polish (Day 7)
- [ ] Add docstrings to all public methods
- [ ] Create usage examples
- [ ] Update README with new feature
- [ ] Final testing and bug fixes

## Non-Goals (Out of Scope)

- ❌ Implementing hyperparameter evolution itself (Phase 3)
- ❌ Plateau detection (Phase 2)
- ❌ Cloud execution (Future roadmap)
- ❌ Advanced ML-based strategy prediction

## Future Enhancements (Post-Phase 1)

- Add confidence threshold for auto-selection (skip user confirmation if >95%)
- Multi-language support for user messages
- Telemetry for strategy success across all users
- A/B testing different recommendation algorithms

## Dependencies

**Python Packages** (add to `pyproject.toml`):
```toml
[project.dependencies]
# (existing dependencies)
prompt-toolkit = "^3.0.0"  # For interactive user prompts
```

**SDK Components**:
- Memory system (existing)
- Agent SDK (existing)

## Migration Notes

**Breaking Changes**: None - this is a new feature that enhances existing flow

**Backward Compatibility**:
- If user doesn't want strategy selection, can disable via config:
  ```json
  {
    "strategy_selection": {
      "enabled": false,
      "auto_select": false
    }
  }
  ```

## Questions & Decisions

**Q**: Should we auto-select if confidence is very high (>95%)?
**A**: No, always ask user in Phase 1. Can add auto-select in future phase.

**Q**: How many past runs needed to influence recommendations?
**A**: Require minimum 3 samples with same strategy+keywords before boosting confidence.

**Q**: What if user picks "custom" option?
**A**: Show advanced configuration menu (implement in Phase 1).

## Rollout Plan

1. **Week 1**: Implement and test internally
2. **Week 2**: Beta release to showcase projects
3. **Week 3**: Collect feedback, iterate
4. **Week 4**: Stable release

## Notes

- Keep UI messages concise and actionable
- Use emoji sparingly (only for status indicators)
- All user-facing text should be clear to non-experts
- Memory queries should be fast (<100ms)
