# Phase 3: Hyperparameter Evolution

**Status**: 🔴 Not Started
**Priority**: P0
**Estimated Duration**: 1 week
**Dependencies**: Phase 1 (Strategy Selection), Phase 2 (Plateau Detection)

## Overview

Implement the ability to evolve algorithm hyperparameters using black-box optimization. This was the critical missing piece in the Santa challenge - the sparrow algorithm with default parameters performed poorly, but with optimized parameters it likely would have won. The system will:
- Detect tunable parameters in code using Claude
- Extract parameters and infer reasonable bounds
- Use CMA-ES, Bayesian optimization, or random search
- Validate improvements with trust system
- Apply optimized parameters back to code
- Store trials in memory for learning

## Goals

- ✅ Automatic parameter detection in any code
- ✅ Multiple optimization methods (CMA-ES, Bayesian, random)
- ✅ Trust system validation of improvements
- ✅ Intelligent parameter bound inference
- ✅ Code transformation to apply optimized parameters
- ✅ Memory storage of all trials and best parameters

## Architecture Components

### 1. Hyperparameter Detector
**File**: `evolve_sdk/hyperparameter/detector.py`

Analyzes code to find tunable parameters:
- Uses Claude to identify parameters in code
- Extracts parameter types, current values, locations
- Infers reasonable bounds from context
- Uses adversary to validate extraction correctness

**Key Methods**:
```python
def detect_parameters(solution: Solution) -> ParameterSpace
def _analyze_with_claude(code: str) -> dict
def _infer_bounds(param: dict) -> Tuple
def _validate_extraction(code: str, params: List) -> bool
```

**Detection Patterns**:
- Constructor parameters: `def __init__(self, lr=0.01, iters=1000)`
- Class constants: `LEARNING_RATE = 0.01`
- Magic numbers: `for i in range(100)`, `threshold = 0.5`
- Configuration dicts: `config = {'temperature': 1.0, 'steps': 5000}`

### 2. Hyperparameter Evolver
**File**: `evolve_sdk/hyperparameter/evolver.py`

Optimizes parameters using black-box methods:
- Creates objective function from user's evaluator
- Runs optimization (CMA-ES, Bayesian, or random search)
- Tracks all trials in memory
- Validates improvements with trust system
- Returns best parameters found

**Key Methods**:
```python
def evolve(solution: Solution, param_space: ParameterSpace,
          evaluate: Callable, budget: int, method: str) -> HyperparameterResult
def _cmaes_optimize(...) -> HyperparameterResult
def _bayesian_optimize(...) -> HyperparameterResult
def _random_search(...) -> HyperparameterResult
def _apply_parameters(solution: Solution, params: Dict) -> Solution
```

### 3. Search Methods
**File**: `evolve_sdk/hyperparameter/search_methods.py`

Abstract interface for different search algorithms:
```python
class SearchMethod(ABC):
    @abstractmethod
    def search(objective: Callable, initial: Any, budget: int) -> Any

class CMAESSearch(SearchMethod):
    """CMA-ES optimization - best for continuous params."""

class BayesianSearch(SearchMethod):
    """Bayesian optimization - good for expensive evaluations."""

class RandomSearch(SearchMethod):
    """Random search baseline - surprisingly effective."""
```

### 4. Parameter Space
**File**: `evolve_sdk/hyperparameter/types.py`

Data structures for parameters:
```python
@dataclass
class Parameter:
    name: str
    type: str  # 'int' | 'float' | 'bool' | 'categorical'
    current_value: Any
    bounds: Tuple[Any, Any]
    location: str  # "line 15, __init__ parameter"
    purpose: str   # "Controls step size for optimization"

@dataclass
class ParameterSpace:
    parameters: List[Parameter]
    confidence: float

    def to_vector(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to flat vector for optimization."""

    def from_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """Convert vector back to parameter dict."""

@dataclass
class HyperparameterResult:
    best_parameters: Dict[str, Any]
    best_solution: Solution
    best_fitness: float
    improvement: float
    evaluations_used: int
    method: str
    all_trials: List[Dict]  # For analysis
```

## Integration Points

### With Trust System
**What**: Validate hyperparameter improvements are real

**Where**: `evolve_sdk/trust/validators.py`

**New Validator**:
```python
def validate_hyperparameter_improvement(
    original_solution: Solution,
    improved_solution: Solution,
    claimed_improvement: float,
    parameter_changes: Dict[str, Any]
) -> ValidationResult:
    """
    Check if improvement is genuine:
    - Not from evaluator exploitation
    - Reproducible across multiple runs
    - Parameter changes are reasonable
    """
```

### With Adversary Agent
**What**: Validate parameter extraction is correct

**Where**: `evolve_sdk/agents/adversary.py`

**New Method**:
```python
def validate_parameter_extraction(
    code: str,
    extracted_params: List[Parameter]
) -> ValidationResult:
    """
    Verify:
    - All tunable parameters were found
    - No false positives (non-parameters detected)
    - Bounds are reasonable
    """
```

### With Memory System
**What**: Store all hyperparameter trials and best configs

**Where**: `evolve_sdk/memory/schemas.py`

**New Frame Types**:
```python
{
    "frame_type": "hyperparameter_trial",
    "metadata": {
        "solution_id": "...",
        "parameters": {"lr": 0.05, "iters": 2000},
        "fitness": 0.85,
        "trial_number": 42
    }
}

{
    "frame_type": "hyperparameter_result",
    "metadata": {
        "original_fitness": 0.75,
        "improved_fitness": 0.92,
        "improvement": 0.227,
        "best_parameters": {"lr": 0.037, "iters": 1523},
        "method": "cmaes",
        "evaluations_used": 87
    }
}
```

### With Main Evolution Loop
**What**: Execute hyperparameter strategy when selected

**Where**: `evolve_sdk/evolution.py`

**Changes**:
```python
def _execute_strategy(self, strategy: str, solution: Solution, generation: int):
    if strategy == "hyperparameter_evolution":
        return self._hyperparameter_evolution(solution, generation)
    ...

def _hyperparameter_evolution(self, solution: Solution, generation: int):
    # Detect parameters
    param_space = self.param_detector.detect_parameters(solution)

    if not param_space.parameters:
        # Fall back to code evolution
        return self._code_evolution(solution, generation)

    # Evolve parameters
    result = self.param_evolver.evolve(
        solution=solution,
        param_space=param_space,
        evaluate=self.evaluation_agent.evaluate,
        budget=100,
        method="cmaes"
    )

    return result.best_solution
```

## Files to Create

```
sdk/evolve_sdk/hyperparameter/
├── __init__.py
├── detector.py               # Parameter detection
├── evolver.py                # Hyperparameter optimization
├── search_methods.py         # Optimization algorithms
└── types.py                  # Data structures

sdk/tests/hyperparameter/
├── __init__.py
├── test_detector.py
├── test_evolver.py
├── test_search_methods.py
└── test_integration.py
```

## Files to Modify

- `evolve_sdk/agents/adversary.py` - Add validate_parameter_extraction()
- `evolve_sdk/trust/validators.py` - Add hyperparameter improvement validator
- `evolve_sdk/memory/schemas.py` - Add hyperparameter frame types
- `evolve_sdk/evolution.py` - Add _hyperparameter_evolution() method
- `pyproject.toml` - Add dependencies (cma, scikit-optimize)

## Validation Tests

### Test 1: Parameter Detection
```python
def test_parameter_detection():
    """Parameters correctly detected in code."""
    code = '''
class SparrowSolver:
    def __init__(self, exploration_iters=15000, base_penalty=1.0):
        self.exploration_iters = exploration_iters
        self.base_penalty = base_penalty
'''

    solution = Solution(code=code)
    detector = HyperparameterDetector(adversary, memory)
    param_space = detector.detect_parameters(solution)

    assert len(param_space.parameters) == 2
    assert any(p.name == "exploration_iters" for p in param_space.parameters)
    assert any(p.name == "base_penalty" for p in param_space.parameters)

    # Check bounds inference
    exploration = [p for p in param_space.parameters if p.name == "exploration_iters"][0]
    assert exploration.bounds[0] < 15000
    assert exploration.bounds[1] > 15000
```

### Test 2: Bound Inference
```python
def test_bound_inference():
    """Bounds correctly inferred from parameter types/values."""
    detector = HyperparameterDetector(adversary, memory)

    # Integer parameter
    bounds_int = detector._infer_bounds({
        'type': 'int',
        'current_value': 100
    })
    assert bounds_int == (10, 1000)  # 10x range

    # Float parameter (0-1 range)
    bounds_float = detector._infer_bounds({
        'type': 'float',
        'current_value': 0.05
    })
    assert bounds_float[0] < 0.05 < bounds_float[1]
    assert bounds_float[1] <= 1.0  # Capped at 1.0
```

### Test 3: CMA-ES Optimization
```python
def test_cmaes_optimization():
    """CMA-ES finds optimal parameters."""
    # Simple optimization: tune a*x + b to fit y = 2*x + 3
    code = '''
def linear_fn(x, a=1.0, b=0.0):
    return a * x + b
'''

    solution = Solution(code=code, fitness=0.5)
    param_space = detector.detect_parameters(solution)

    def evaluate(sol):
        exec(sol.code)
        predictions = [linear_fn(x) for x in range(10)]
        targets = [2*x + 3 for x in range(10)]
        mse = sum((p - t)**2 for p, t in zip(predictions, targets)) / len(predictions)
        return -mse

    result = evolver.evolve(
        solution=solution,
        param_space=param_space,
        evaluate=evaluate,
        budget=50,
        method="cmaes"
    )

    # Should find a≈2, b≈3
    assert abs(result.best_parameters['a'] - 2.0) < 0.1
    assert abs(result.best_parameters['b'] - 3.0) < 0.1
    assert result.improvement > 0.9
```

### Test 4: Trust System Validation
```python
def test_trust_validates_improvements():
    """Trust system catches suspicious hyperparameter improvements."""
    solution = Solution(code="...", fitness=1.0)

    # Claim 100x improvement (suspicious!)
    result = HyperparameterResult(
        best_fitness=100.0,
        improvement=99.0,
        best_parameters={"lr": 999.0}  # Suspicious value
    )

    validation = trust_system.validate_hyperparameter_improvement(
        original_solution=solution,
        improved_solution=result.best_solution,
        claimed_improvement=99.0,
        parameter_changes=result.best_parameters
    )

    assert validation.is_trustworthy == False
    assert "suspicious" in validation.warning.lower()
```

### Test 5: Parameter Application
```python
def test_apply_parameters_to_code():
    """Optimized parameters correctly applied to code."""
    original_code = '''
class Solver:
    def __init__(self, lr=0.01, iters=100):
        self.lr = lr
        self.iters = iters
'''

    solution = Solution(code=original_code)
    new_params = {"lr": 0.05, "iters": 500}

    evolver = HyperparameterEvolver(memory, trust, ui)
    updated_solution = evolver._apply_parameters(solution, new_params)

    # Check new code has updated values
    assert "lr=0.05" in updated_solution.code or "0.05" in updated_solution.code
    assert "iters=500" in updated_solution.code or "500" in updated_solution.code

    # Verify code still runs
    exec(updated_solution.code)
```

### Test 6: Memory Storage
```python
def test_hyperparameter_trials_stored():
    """All trials stored in memory for analysis."""
    result = evolver.evolve(
        solution=solution,
        param_space=param_space,
        evaluate=evaluate,
        budget=20,
        method="random"
    )

    # Check memory has trial records
    trials = memory.query(frame_type="hyperparameter_trial")
    assert len(trials) == 20  # All trials stored

    # Check result record
    results = memory.query(frame_type="hyperparameter_result")
    assert len(results) == 1
    assert results[0].metadata['improvement'] == result.improvement
```

## Success Criteria

- [x] Detects parameters in various code patterns (constructors, constants, magic numbers)
- [x] Infers reasonable bounds for int, float, categorical parameters
- [x] CMA-ES successfully optimizes simple test functions
- [x] Bayesian optimization works for expensive evaluations
- [x] Trust system validates improvements and catches exploits
- [x] Parameters correctly applied back to code
- [x] All trials stored in memory
- [x] User confirmation requested with budget estimates
- [x] All tests pass
- [x] No regressions

## Implementation Checklist

### Step 1: Types & Infrastructure (Day 1)
- [ ] Create `hyperparameter/types.py` with Parameter, ParameterSpace, etc.
- [ ] Implement `to_vector()` and `from_vector()` for optimization
- [ ] Add dependencies to pyproject.toml (cma, scikit-optimize)
- [ ] Write unit tests for types

### Step 2: Parameter Detector (Day 2-3)
- [ ] Create `detector.py`
- [ ] Implement `_analyze_with_claude()` with comprehensive prompt
- [ ] Implement `_infer_bounds()` for different parameter types
- [ ] Add adversary validation hook
- [ ] Test detection on various code patterns

### Step 3: Search Methods (Day 3-4)
- [ ] Create `search_methods.py` with abstract base
- [ ] Implement `CMAESSearch` using `cma` library
- [ ] Implement `BayesianSearch` using `skopt`
- [ ] Implement `RandomSearch` baseline
- [ ] Test each method on simple optimization problems

### Step 4: Hyperparameter Evolver (Day 4-5)
- [ ] Create `evolver.py`
- [ ] Implement `evolve()` orchestration method
- [ ] Implement `_apply_parameters()` using Claude for code transformation
- [ ] Add trust system validation
- [ ] Add user confirmation for budget
- [ ] Test on real code examples

### Step 5: Integration (Day 6)
- [ ] Add adversary parameter validation method
- [ ] Add trust system hyperparameter validator
- [ ] Add memory frame types
- [ ] Integrate into evolution loop
- [ ] Write integration tests

### Step 6: Testing & Polish (Day 7)
- [ ] End-to-end testing with real algorithms
- [ ] Test on santa-2025-packing sparrow algorithm
- [ ] Edge case handling (no params found, budget too small, etc.)
- [ ] Documentation and examples
- [ ] Performance optimization

## Non-Goals (Out of Scope)

- ❌ Multi-fidelity optimization (use cheap approximations first)
- ❌ Gradient-based optimization (code must be differentiable)
- ❌ Transfer learning between problems
- ❌ Automatic bound discovery through probing

## Configuration Options

Add to `evolve_config.json`:
```json
{
  "hyperparameter_evolution": {
    "enabled": true,
    "default_method": "cmaes",
    "default_budget": 100,
    "ask_user_confirmation": true,
    "min_parameters_required": 1,
    "adversary_validation": true,
    "trust_validation": true,
    "store_all_trials": true
  }
}
```

## Edge Cases to Handle

1. **No parameters found**: Fall back to code evolution, warn user
2. **Budget too small**: Warn user, suggest minimum (e.g., 20 * n_params)
3. **Categorical parameters**: One-hot encode for optimization
4. **Dependent parameters**: Document limitation, future enhancement
5. **Parameter application fails**: Keep original code, log error
6. **All trials invalid**: Return original solution unchanged

## Future Enhancements (Post-Phase 3)

- Multi-fidelity optimization (cheap approximation → full evaluation)
- Parallel trials for faster optimization
- Automatic budget selection based on parameter count
- Transfer learning from similar problems
- Gradient-based optimization for differentiable code
- Conditional parameter spaces (parameter B only matters if parameter A > threshold)

## Dependencies

**SDK Components**:
- Phase 1 (Strategy Selection) - REQUIRED
- Phase 2 (Plateau Detection) - OPTIONAL (enhances workflow)
- Trust system - REQUIRED
- Adversary agent - REQUIRED
- Memory system - REQUIRED

**Python Packages** (add to `pyproject.toml`):
```toml
[project.dependencies]
cma = "^3.3.0"              # CMA-ES optimization
scikit-optimize = "^0.9.0"  # Bayesian optimization
numpy = "^1.24.0"           # Numerical operations
scipy = "^1.10.0"           # Scientific computing
```

## Migration Notes

**Breaking Changes**: None

**Backward Compatibility**:
- Hyperparameter evolution is a new strategy, doesn't affect existing code evolution
- Can be disabled via config

## Real-World Validation

**Test on Santa 2025 Sparrow Algorithm**:
```python
# This should find much better parameters than defaults
solution = load_santa_sparrow_algorithm()  # Gen100 implementation

result = evolver.evolve(
    solution=solution,
    param_space=detector.detect_parameters(solution),
    evaluate=lambda sol: evaluate_santa_packing(sol, n_range=(5, 20)),
    budget=200,
    method="cmaes"
)

# Should find:
# - exploration_iters: ~25000 (not 15000)
# - compression_iters: ~50000 (not 30000)
# - base_penalty: ~2.5 (not 1.0)
# etc.

# Expected improvement: >10% on santa challenge
```

## Rollout Plan

1. **Week 1**: Implementation and unit testing
2. **Week 2**: Integration testing on santa-2025-packing
3. **Week 3**: Beta test on other showcases
4. **Week 4**: Collect feedback, iterate, release

## Notes

- Parameter detection is the hardest part - be thorough
- CMA-ES works best for continuous parameters (4-20 dimensions)
- Bayesian optimization better for expensive evaluations (<100 budget)
- Always store ALL trials for post-hoc analysis
- Trust validation critical - users will try to game the system
