# Implementation Order - Master Checklist

This document provides a linear, step-by-step implementation order for all 4 phases. Use this as your guide after context clears.

**Total Duration**: ~3 weeks (15 working days)

---

## Pre-Implementation Setup

### Dependencies to Add
Add to `pyproject.toml`:
```toml
[project.dependencies]
# Phase 1
prompt-toolkit = "^3.0.0"  # Interactive prompts

# Phase 3
cma = "^3.3.0"              # CMA-ES optimization
scikit-optimize = "^0.9.0"  # Bayesian optimization
numpy = "^1.24.0"
scipy = "^1.10.0"
```

---

## PHASE 1: Strategy Selection (Week 1, Days 1-7)

### Day 1: Foundation & Types

#### 1.1 Create Directory Structure
```bash
mkdir -p evolve_sdk/strategy
mkdir -p evolve_sdk/ui
mkdir -p tests/strategy
mkdir -p tests/ui
```

#### 1.2 Create Base Types
**File**: `evolve_sdk/strategy/__init__.py`
```python
"""Strategy selection and orchestration."""
from .types import (
    ProblemContext,
    StrategyRecommendation,
    StrategyDecision
)
from .problem_analyzer import ProblemAnalyzer
from .strategy_selector import StrategySelector

__all__ = [
    "ProblemContext",
    "StrategyRecommendation",
    "StrategyDecision",
    "ProblemAnalyzer",
    "StrategySelector"
]
```

#### 1.3 Create Types Module
**File**: `evolve_sdk/strategy/types.py` (~100 lines)
```python
"""Data structures for strategy selection."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ProblemContext:
    """Context about the problem being solved."""
    problem_type: str  # "new_algorithm" | "tune_existing" | "stuck_evolution" | "hybrid"
    has_parameters: bool
    complexity: str  # "simple" | "medium" | "complex"
    similar_past_runs: List[Any]  # List[MemoryFrame]
    is_continuation: bool

@dataclass
class StrategyRecommendation:
    """A recommended strategy with confidence."""
    strategy: str
    confidence: float
    reason: str

@dataclass
class StrategyDecision:
    """Final strategy decision."""
    strategy: str
    auto_selected: bool
    user_confirmed: bool
```

**Test**: `tests/strategy/test_types.py` (~50 lines)
- Test dataclass construction
- Test serialization

---

### Day 2: Problem Analyzer

#### 2.1 Create Problem Analyzer
**File**: `evolve_sdk/strategy/problem_analyzer.py` (~200 lines)
```python
"""Analyze problems to determine optimization strategy."""
from typing import Optional
from ..memory.memory_manager import MemoryManager
from .types import ProblemContext

class ProblemAnalyzer:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def analyze(self,
                user_prompt: str,
                current_solution: Optional[Any] = None) -> ProblemContext:
        """Analyze the problem and return context."""
        # Implementation

    def _detect_parameters(self, solution: Any) -> bool:
        """Use Claude to detect if code has parameters."""
        # Implementation

    def _classify_problem_type(self, prompt: str, is_continuation: bool) -> str:
        """Classify problem type from prompt."""
        # Implementation
```

**Test**: `tests/strategy/test_problem_analyzer.py` (~150 lines)
- `test_new_algorithm_classification()` - New algorithm detected
- `test_tune_existing_classification()` - Parameter tuning detected
- `test_parameter_detection_in_code()` - Parameters found in code
- `test_continuation_classification()` - Resume detected

---

### Day 3: Strategy Selector (Part 1)

#### 3.1 Create User Interface Module
**File**: `evolve_sdk/ui/__init__.py`
```python
"""User interface components."""
from .user_interface import UserInterface

__all__ = ["UserInterface"]
```

**File**: `evolve_sdk/ui/user_interface.py` (~150 lines)
```python
"""User interaction methods."""
from typing import List, Dict, Any

class UserInterface:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive

    def ask_user_choice(self,
                       message: str,
                       options: List[Dict[str, str]],
                       default: str) -> str:
        """Ask user to choose from options."""
        # Implementation using prompt-toolkit

    def show_info(self, message: str):
        """Show info message."""

    def show_warning(self, message: str):
        """Show warning message."""

    def show_success(self, message: str):
        """Show success message."""
```

**Test**: `tests/ui/test_user_interface.py` (~100 lines)
- `test_mock_ui_auto_select()` - Mock UI for testing
- `test_user_choice_with_options()` - Choice selection
- `test_message_display()` - Message formatting

---

### Day 4: Strategy Selector (Part 2)

#### 4.1 Create Strategy Selector
**File**: `evolve_sdk/strategy/strategy_selector.py` (~250 lines)
```python
"""Select and recommend strategies."""
from typing import List
from ..memory.memory_manager import MemoryManager
from ..ui.user_interface import UserInterface
from .types import ProblemContext, StrategyRecommendation, StrategyDecision

class StrategySelector:
    def __init__(self, memory: MemoryManager, ui: UserInterface):
        self.memory = memory
        self.ui = ui

    def select(self, context: ProblemContext) -> StrategyDecision:
        """Select strategy based on context."""
        recommendations = self._recommend_strategies(context)
        recommendations = self._enhance_with_memory(recommendations, context)
        return self._ask_user(recommendations, context)

    def _recommend_strategies(self, context: ProblemContext) -> List[StrategyRecommendation]:
        """Generate recommendations."""
        # Implementation

    def _enhance_with_memory(self,
                            recommendations: List[StrategyRecommendation],
                            context: ProblemContext) -> List[StrategyRecommendation]:
        """Enhance with historical data."""
        # Implementation

    def _ask_user(self,
                  recommendations: List[StrategyRecommendation],
                  context: ProblemContext) -> StrategyDecision:
        """Ask user for confirmation."""
        # Implementation
```

**Test**: `tests/strategy/test_strategy_selector.py` (~200 lines)
- `test_recommend_code_evolution_for_new()` - New algorithm → code evolution
- `test_recommend_hyperparameter_for_params()` - Has params → hyperparameter
- `test_memory_enhances_confidence()` - Historical success boosts confidence
- `test_user_confirmation_flow()` - User asked and confirms

---

### Day 5: Memory Integration

#### 5.1 Extend Memory Queries
**File**: `evolve_sdk/memory/queries.py` (modify, add ~100 lines)

Add methods:
```python
def query_similar_problems(self, prompt: str, limit: int = 10) -> List[MemoryFrame]:
    """Find similar past problems using embeddings."""
    # Implementation

def query_strategy_success_rate(self, strategy: str, keywords: List[str]) -> float:
    """Calculate success rate for strategy on similar problems."""
    # Implementation
```

#### 5.2 Extend Memory Schemas
**File**: `evolve_sdk/memory/schemas.py` (modify, add ~50 lines)

Add frame type:
```python
FRAME_TYPES = {
    # ... existing ...
    "strategy_decision": {
        "description": "Strategy selection decision",
        "required_fields": ["strategy", "problem_type", "user_confirmed"]
    }
}
```

**Test**: `tests/memory/test_strategy_queries.py` (~100 lines)
- `test_query_similar_problems()` - Similarity search works
- `test_strategy_success_rate()` - Success rate calculation

---

### Day 6: Integration with Main Loop

#### 6.1 Create Runner Integration Points
**File**: `evolve_sdk/runner.py` (modify, add ~50 lines at start of run())

Add at start of `run()` method:
```python
# PHASE 1: Strategy selection
from .strategy import ProblemAnalyzer, StrategySelector
from .ui import UserInterface

analyzer = ProblemAnalyzer(self.memory)
ui = UserInterface(interactive=not self.config.batch_mode)
selector = StrategySelector(self.memory, ui)

context = analyzer.analyze(self.problem, self.current_solution)
decision = selector.select(context)
current_strategy = decision.strategy

logger.info(f"Selected strategy: {current_strategy}")
```

**Test**: `tests/integration/test_phase1_integration.py` (~150 lines)
- `test_end_to_end_strategy_selection()` - Full flow works
- `test_strategy_stored_in_memory()` - Decision recorded

---

### Day 7: Testing & Documentation

#### 7.1 Run Full Test Suite
```bash
pytest tests/strategy/ -v
pytest tests/ui/ -v
pytest tests/integration/test_phase1_integration.py -v
```

#### 7.2 Create Examples
**File**: `examples/phase1_demo.py` (~50 lines)
```python
"""Demo of Phase 1 strategy selection."""
# Example showing analyzer and selector in action
```

#### 7.3 Update Documentation
**File**: `README.md` (add Phase 1 section)

---

## PHASE 2: Plateau Detection (Week 2, Days 8-11)

### Day 8: Plateau Detector

#### 8.1 Extend Strategy Types
**File**: `evolve_sdk/strategy/types.py` (add ~50 lines)
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

#### 8.2 Create Plateau Detector
**File**: `evolve_sdk/strategy/plateau_detector.py` (~200 lines)
```python
"""Detect when evolution is stuck."""
from typing import List
from ..memory.memory_manager import MemoryManager
from ..agents.adversary import AdversaryAgent
from .types import PlateauDetection

class PlateauDetector:
    def __init__(self,
                 memory: MemoryManager,
                 adversary: AdversaryAgent,
                 patience: int = 20,
                 min_improvement: float = 0.01):
        self.memory = memory
        self.adversary = adversary
        self.patience = patience
        self.min_improvement = min_improvement

    def check_plateau(self, generation: int) -> PlateauDetection:
        """Check if plateaued."""
        # Implementation

    def _check_mutation_diversity(self, recent_gens: List) -> float:
        """Calculate mutation diversity."""
        # Implementation

    def _suggest_actions(self, improvement: float, diversity: float) -> List[str]:
        """Suggest what to do."""
        # Implementation
```

**File**: `evolve_sdk/strategy/__init__.py` (update exports)

**Test**: `tests/strategy/test_plateau_detector.py` (~150 lines)
- `test_plateau_detected_after_patience()` - Triggers after N gens
- `test_adversary_validates_plateau()` - Adversary can reject
- `test_mutation_diversity_calculation()` - Diversity computed correctly

---

### Day 9: Adversary Extension & Auto Switcher

#### 9.1 Extend Adversary Agent
**File**: `evolve_sdk/agents/adversary.py` (add ~100 lines)

Add method:
```python
def validate_plateau(self,
                    recent_generations: List[Any],
                    claimed_plateau: bool) -> ValidationResult:
    """Validate plateau is real, not evaluator bug."""
    # Implementation
```

**Test**: `tests/agents/test_adversary_plateau.py` (~100 lines)
- `test_adversary_validates_plateau()` - Validation works
- `test_adversary_detects_evaluator_bug()` - Can reject false plateaus

#### 9.2 Create Auto Switcher
**File**: `evolve_sdk/strategy/auto_switcher.py` (~200 lines)
```python
"""Automatic strategy switching."""
from typing import Optional, List
from .plateau_detector import PlateauDetector
from .strategy_selector import StrategySelector
from ..ui.user_interface import UserInterface
from ..memory.memory_manager import MemoryManager
from .types import StrategySwitch

class AutoStrategySwitcher:
    def __init__(self,
                 detector: PlateauDetector,
                 selector: StrategySelector,
                 ui: UserInterface,
                 memory: MemoryManager):
        self.detector = detector
        self.selector = selector
        self.ui = ui
        self.memory = memory

    def check_and_switch(self,
                        generation: int,
                        current_strategy: str) -> Optional[StrategySwitch]:
        """Check for plateau and switch if needed."""
        # Implementation

    def _filter_tried_strategies(self, actions: List[str]) -> List[str]:
        """Remove already-tried strategies."""
        # Implementation

    def _propose_switch(self, ...) -> Optional[StrategySwitch]:
        """Ask user about switching."""
        # Implementation
```

**Test**: `tests/strategy/test_auto_switcher.py` (~150 lines)
- `test_filters_tried_strategies()` - No duplicates
- `test_asks_user_before_switch()` - User confirmation
- `test_user_can_decline_switch()` - User says no

---

### Day 10: Memory & Integration

#### 10.1 Extend Memory Schemas
**File**: `evolve_sdk/memory/schemas.py` (add ~30 lines)
```python
"strategy_switch": {
    "description": "Strategy switch event",
    "required_fields": ["generation", "from_strategy", "to_strategy", "reason"]
}
```

#### 10.2 Integrate into Runner
**File**: `evolve_sdk/runner.py` (modify main loop, add ~40 lines)

Add to generation loop:
```python
# PHASE 2: Check for plateau
plateau_detector = PlateauDetector(self.memory, self.adversary)
auto_switcher = AutoStrategySwitcher(plateau_detector, selector, ui, self.memory)

for generation in range(max_generations):
    # Check plateau
    switch = auto_switcher.check_and_switch(generation, current_strategy)
    if switch:
        logger.info(f"Switching: {switch.from_strategy} → {switch.to_strategy}")
        current_strategy = switch.to_strategy
        self.memory.store_strategy_switch(switch)

    # Continue evolution...
```

**Test**: `tests/integration/test_phase2_integration.py` (~150 lines)
- `test_plateau_triggers_switch()` - Auto-switch works
- `test_switch_stored_in_memory()` - Recorded properly

---

### Day 11: Testing & Documentation

#### 11.1 Full Test Suite
```bash
pytest tests/strategy/test_plateau_detector.py -v
pytest tests/strategy/test_auto_switcher.py -v
pytest tests/agents/test_adversary_plateau.py -v
pytest tests/integration/test_phase2_integration.py -v
```

#### 11.2 Update Documentation

---

## PHASE 3: Hyperparameter Evolution (Week 3, Days 12-18)

### Day 12: Types & Infrastructure

#### 12.1 Create Directory
```bash
mkdir -p evolve_sdk/hyperparameter
mkdir -p tests/hyperparameter
```

#### 12.2 Create Types
**File**: `evolve_sdk/hyperparameter/__init__.py`
```python
"""Hyperparameter evolution."""
from .types import Parameter, ParameterSpace, HyperparameterResult
from .detector import HyperparameterDetector
from .evolver import HyperparameterEvolver

__all__ = [
    "Parameter",
    "ParameterSpace",
    "HyperparameterResult",
    "HyperparameterDetector",
    "HyperparameterEvolver"
]
```

**File**: `evolve_sdk/hyperparameter/types.py` (~200 lines)
```python
"""Data structures for hyperparameter evolution."""
from dataclasses import dataclass
from typing import Any, Tuple, List, Dict
import numpy as np

@dataclass
class Parameter:
    name: str
    type: str  # 'int' | 'float' | 'bool' | 'categorical'
    current_value: Any
    bounds: Tuple[Any, Any]
    location: str
    purpose: str = ""

@dataclass
class ParameterSpace:
    parameters: List[Parameter]
    confidence: float

    def to_vector(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to optimization vector."""
        # Implementation

    def from_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """Convert back from vector."""
        # Implementation

@dataclass
class HyperparameterResult:
    best_parameters: Dict[str, Any]
    best_solution: Any
    best_fitness: float
    improvement: float
    evaluations_used: int
    method: str
    all_trials: List[Dict] = None
```

**Test**: `tests/hyperparameter/test_types.py` (~100 lines)
- `test_parameter_space_to_vector()` - Conversion works
- `test_parameter_space_from_vector()` - Round-trip works

#### 12.3 Install Dependencies
```bash
pip install cma scikit-optimize numpy scipy
```

---

### Day 13: Search Methods

#### 13.1 Create Search Methods
**File**: `evolve_sdk/hyperparameter/search_methods.py` (~300 lines)
```python
"""Black-box optimization methods."""
from abc import ABC, abstractmethod
from typing import Callable, Any
import cma
from skopt import gp_minimize

class SearchMethod(ABC):
    @abstractmethod
    def search(self, objective: Callable, initial: Any, budget: int) -> Any:
        pass

class CMAESSearch(SearchMethod):
    """CMA-ES optimization."""
    def search(self, objective, initial, budget):
        # Implementation using cma library
        pass

class BayesianSearch(SearchMethod):
    """Bayesian optimization."""
    def search(self, objective, initial, budget):
        # Implementation using skopt
        pass

class RandomSearch(SearchMethod):
    """Random search baseline."""
    def search(self, objective, initial, budget):
        # Implementation
        pass
```

**Test**: `tests/hyperparameter/test_search_methods.py` (~200 lines)
- `test_cmaes_optimizes_simple_function()` - CMA-ES works
- `test_bayesian_optimizes_function()` - Bayesian works
- `test_random_search_baseline()` - Random works

---

### Day 14-15: Parameter Detector

#### 14.1 Create Detector
**File**: `evolve_sdk/hyperparameter/detector.py` (~250 lines)
```python
"""Detect tunable parameters in code."""
from typing import Any
from ..agents.adversary import AdversaryAgent
from ..memory.memory_manager import MemoryManager
from .types import ParameterSpace, Parameter

class HyperparameterDetector:
    def __init__(self,
                 adversary: AdversaryAgent,
                 memory: MemoryManager):
        self.adversary = adversary
        self.memory = memory

    def detect_parameters(self, solution: Any) -> ParameterSpace:
        """Analyze code and find parameters."""
        analysis = self._analyze_with_claude(solution.code)

        # Validate with adversary
        validation = self.adversary.validate_parameter_extraction(
            solution.code,
            analysis['parameters']
        )

        if not validation.is_valid:
            # Handle extraction failure
            pass

        # Build parameter space
        param_space = self._build_parameter_space(analysis)

        # Store in memory
        self.memory.store_parameter_detection(param_space)

        return param_space

    def _analyze_with_claude(self, code: str) -> dict:
        """Use Claude to find parameters."""
        # Implementation with comprehensive prompt
        pass

    def _infer_bounds(self, param: dict) -> Tuple:
        """Infer bounds from parameter type and value."""
        # Implementation
        pass
```

#### 14.2 Extend Adversary
**File**: `evolve_sdk/agents/adversary.py` (add ~80 lines)

Add method:
```python
def validate_parameter_extraction(self,
                                 code: str,
                                 extracted_params: List[Parameter]) -> ValidationResult:
    """Validate parameter extraction correctness."""
    # Implementation
```

**Test**: `tests/hyperparameter/test_detector.py` (~200 lines)
- `test_detects_constructor_params()` - __init__ params found
- `test_detects_class_constants()` - Constants found
- `test_detects_magic_numbers()` - Magic numbers found
- `test_infers_bounds_correctly()` - Bounds reasonable
- `test_adversary_validates()` - Adversary checks extraction

---

### Day 16-17: Hyperparameter Evolver

#### 16.1 Create Evolver
**File**: `evolve_sdk/hyperparameter/evolver.py` (~350 lines)
```python
"""Evolve hyperparameters using optimization."""
from typing import Callable
from ..memory.memory_manager import MemoryManager
from ..trust.validators import TrustValidator
from ..ui.user_interface import UserInterface
from .types import ParameterSpace, HyperparameterResult
from .search_methods import CMAESSearch, BayesianSearch, RandomSearch

class HyperparameterEvolver:
    def __init__(self,
                 memory: MemoryManager,
                 trust_validator: TrustValidator,
                 ui: UserInterface):
        self.memory = memory
        self.trust = trust_validator
        self.ui = ui

    def evolve(self,
               solution: Any,
               param_space: ParameterSpace,
               evaluate: Callable,
               budget: int = 100,
               method: str = "cmaes") -> HyperparameterResult:
        """Optimize hyperparameters."""

        # Ask user confirmation
        if not self._confirm_budget(param_space, budget):
            return None

        # Select search method
        search = self._get_search_method(method)

        # Run optimization
        result = self._run_optimization(search, solution, param_space, evaluate, budget)

        # Validate with trust system
        if result.improvement > 0:
            validation = self.trust.validate_hyperparameter_improvement(
                solution,
                result.best_solution,
                result.improvement,
                result.best_parameters
            )
            if not validation.is_trustworthy:
                # Handle untrusted result
                pass

        # Store in memory
        self.memory.store_hyperparameter_result(result)

        return result

    def _run_optimization(self, ...) -> HyperparameterResult:
        """Execute optimization."""
        # Implementation

    def _apply_parameters(self, solution: Any, params: Dict) -> Any:
        """Apply parameters to code using Claude."""
        # Implementation
```

#### 16.2 Extend Trust System
**File**: `evolve_sdk/trust/validators.py` (add ~100 lines)

Add validator:
```python
def validate_hyperparameter_improvement(self,
                                       original: Any,
                                       improved: Any,
                                       improvement: float,
                                       params: Dict) -> ValidationResult:
    """Validate hyperparameter improvements."""
    # Implementation
```

**Test**: `tests/hyperparameter/test_evolver.py` (~250 lines)
- `test_cmaes_optimization_simple()` - Optimizes simple function
- `test_bayesian_optimization()` - Bayesian method works
- `test_apply_parameters_to_code()` - Code transformation works
- `test_trust_validates_improvement()` - Trust system catches exploits
- `test_all_trials_stored()` - Memory records all trials

---

### Day 18: Integration & Testing

#### 18.1 Extend Memory Schemas
**File**: `evolve_sdk/memory/schemas.py` (add ~60 lines)
```python
"hyperparameter_trial": {
    "description": "Single hyperparameter trial",
    "required_fields": ["solution_id", "parameters", "fitness", "trial_number"]
},
"hyperparameter_result": {
    "description": "Hyperparameter optimization result",
    "required_fields": ["original_fitness", "improved_fitness", "best_parameters", "method"]
}
```

#### 18.2 Integrate into Runner
**File**: `evolve_sdk/runner.py` (add strategy execution, ~60 lines)

Add method:
```python
def _execute_strategy(self, strategy: str, solution: Any, generation: int) -> Any:
    if strategy == "code_evolution":
        return self._code_evolution(solution, generation)
    elif strategy == "hyperparameter_evolution":
        return self._hyperparameter_evolution(solution, generation)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _hyperparameter_evolution(self, solution: Any, generation: int) -> Any:
    # Detect parameters
    param_space = self.param_detector.detect_parameters(solution)

    if not param_space.parameters:
        # Fall back to code evolution
        return self._code_evolution(solution, generation)

    # Evolve parameters
    result = self.param_evolver.evolve(
        solution=solution,
        param_space=param_space,
        evaluate=self.evaluator.evaluate,
        budget=100,
        method="cmaes"
    )

    return result.best_solution
```

**Test**: `tests/integration/test_phase3_integration.py` (~200 lines)
- `test_hyperparameter_evolution_end_to_end()` - Full flow
- `test_falls_back_to_code_evolution()` - No params → code evolution
- `test_santa_sparrow_optimization()` - Real-world validation

#### 18.3 Full Test Suite
```bash
pytest tests/hyperparameter/ -v
pytest tests/trust/test_hyperparameter_validation.py -v
pytest tests/integration/test_phase3_integration.py -v
```

---

## PHASE 4: Integration & UX (Week 4, Days 19-21)

### Day 19: Orchestrator

#### 19.1 Create Orchestrator
**File**: `evolve_sdk/orchestrator.py` (~400 lines)
```python
"""Main evolution orchestrator."""
from typing import Any, Optional
from .strategy import ProblemAnalyzer, StrategySelector, PlateauDetector, AutoStrategySwitcher
from .hyperparameter import HyperparameterDetector, HyperparameterEvolver
from .memory.memory_manager import MemoryManager
from .agents.adversary import AdversaryAgent
from .trust.validators import TrustValidator
from .ui.user_interface import UserInterface

class EvolutionOrchestrator:
    """Coordinates all evolution strategies and systems."""

    def __init__(self,
                 config: Any,
                 memory: MemoryManager,
                 ui: UserInterface,
                 adversary: AdversaryAgent,
                 trust_validator: TrustValidator):
        self.config = config
        self.memory = memory
        self.ui = ui
        self.adversary = adversary
        self.trust = trust_validator

        # Initialize components
        self.analyzer = ProblemAnalyzer(memory)
        self.selector = StrategySelector(memory, ui)
        self.plateau_detector = PlateauDetector(memory, adversary)
        self.auto_switcher = AutoStrategySwitcher(
            self.plateau_detector, self.selector, ui, memory
        )
        self.param_detector = HyperparameterDetector(adversary, memory)
        self.param_evolver = HyperparameterEvolver(memory, trust_validator, ui)

    def run(self,
            user_prompt: str,
            resume: bool = False) -> EvolutionResult:
        """Main evolution loop."""

        # Load checkpoint if resuming
        if resume:
            state = self.memory.load_checkpoint()
            current_solution = state.champion
            generation = state.generation
        else:
            current_solution = None
            generation = 0

        # PHASE 1: Analyze and select strategy
        context = self.analyzer.analyze(user_prompt, current_solution)
        decision = self.selector.select(context)
        current_strategy = decision.strategy

        self.ui.show_info(f"Starting with strategy: {current_strategy}")

        # Evolution loop
        for gen in range(generation, self.config.max_generations):

            # PHASE 2: Check plateau and auto-switch
            switch = self.auto_switcher.check_and_switch(gen, current_strategy)

            if switch:
                current_strategy = switch.to_strategy
                self.ui.show_strategy_switch(switch)

            # Execute current strategy
            candidate = self._execute_strategy(
                strategy=current_strategy,
                solution=current_solution,
                generation=gen
            )

            # Validate with trust system
            if candidate.fitness > current_solution.fitness:
                validation = self.trust.validate_improvement(...)

                if validation.is_trustworthy:
                    current_solution = candidate
                    self.ui.show_success(...)
                else:
                    self.ui.show_warning(...)

            # Checkpoint
            if gen % 10 == 0:
                self.memory.save_checkpoint(current_solution, gen)

        return EvolutionResult(
            champion=current_solution,
            generations=gen,
            strategies_used=self._get_strategies_used()
        )

    def _execute_strategy(self, strategy: str, solution: Any, generation: int) -> Any:
        """Execute specific strategy."""
        # Implementation
```

**Test**: `tests/test_orchestrator.py` (~200 lines)
- `test_orchestrator_initialization()` - All components created
- `test_run_with_strategy_selection()` - Full flow
- `test_checkpoint_and_resume()` - Resume works

---

### Day 20: Enhanced UI & Error Recovery

#### 20.1 Enhanced UI
**File**: `evolve_sdk/ui/enhanced_ui.py` (~200 lines)
```python
"""Enhanced user interface with progress tracking."""
from .user_interface import UserInterface

class EnhancedUI(UserInterface):
    def show_evolution_start(self, strategy: str, context: Any):
        """Show evolution start banner."""
        # Implementation

    def show_generation_progress(self, gen: int, fitness: float, improvement: float):
        """Show generation progress."""
        # Implementation

    def show_strategy_switch(self, switch: Any):
        """Show strategy switch notification."""
        # Implementation

    def show_hyperparameter_progress(self, trial: int, best_fitness: float):
        """Show hyperparameter optimization progress."""
        # Implementation

    def show_final_summary(self, result: Any):
        """Show final summary with insights."""
        # Implementation
```

#### 20.2 Error Recovery
**File**: `evolve_sdk/recovery.py` (~150 lines)
```python
"""Graceful error handling and recovery."""

class RecoveryManager:
    def handle_adversary_failure(self):
        """Continue without adversary validation."""
        # Implementation

    def handle_parameter_detection_failure(self):
        """Fall back to code evolution."""
        # Implementation

    def handle_optimization_failure(self):
        """Return original solution."""
        # Implementation

    def handle_memory_corruption(self):
        """Continue without history."""
        # Implementation
```

**Test**: `tests/test_recovery.py` (~150 lines)
- `test_graceful_degradation_adversary()` - Works without adversary
- `test_fallback_to_code_evolution()` - Falls back when needed
- `test_memory_corruption_recovery()` - Handles corruption

---

### Day 21: Documentation & Final Validation

#### 21.1 Create Documentation
**File**: `docs/QUICKSTART.md` (~200 lines)
**File**: `docs/STRATEGIES.md` (~300 lines)
**File**: `docs/HYPERPARAMETERS.md` (~400 lines)
**File**: `docs/TROUBLESHOOTING.md` (~200 lines)

#### 21.2 Create Examples
**File**: `examples/simple_optimization.py` (~100 lines)
**File**: `examples/santa_hyperparameter.py` (~150 lines)
**File**: `examples/advanced_strategies.py` (~200 lines)

#### 21.3 Update Main README
**File**: `README.md` (add ~300 lines documenting new features)

#### 21.4 Final Integration Tests
**File**: `tests/integration/test_end_to_end.py` (~300 lines)
- `test_complete_evolution_flow()` - Everything works together
- `test_automatic_strategy_selection()` - Auto-selects correctly
- `test_plateau_and_switch()` - Detects and switches
- `test_hyperparameter_optimization()` - Optimizes parameters
- `test_error_recovery()` - Recovers from failures
- `test_performance_overhead()` - Overhead <10%

#### 21.5 Real-World Validation
```bash
# Test on santa-2025-packing sparrow algorithm
cd ../showcase/santa-2025-packing
python -m evolve_sdk "optimize sparrow algorithm parameters" --config phase3_test.json
```

#### 21.6 Final Test Suite
```bash
pytest tests/ -v --cov=evolve_sdk --cov-report=html
```

---

## Summary: File Count by Phase

### Phase 1: Strategy Selection
- **Create**: 7 new files
- **Modify**: 3 existing files
- **Tests**: 5 test files

### Phase 2: Plateau Detection
- **Create**: 2 new files
- **Modify**: 3 existing files
- **Tests**: 4 test files

### Phase 3: Hyperparameter Evolution
- **Create**: 5 new files
- **Modify**: 4 existing files
- **Tests**: 6 test files

### Phase 4: Integration & UX
- **Create**: 8 new files (including docs/examples)
- **Modify**: 2 existing files
- **Tests**: 4 test files

### Total
- **New files**: 22 implementation + ~15 tests + 7 docs/examples = **44 files**
- **Modified files**: 12 existing files
- **Total tests**: 19 test files with ~2400 lines of tests

---

## Validation Gates

Each phase must pass these gates before moving to the next:

### Phase 1 Gate
```bash
pytest tests/strategy/ tests/ui/ tests/integration/test_phase1_integration.py -v
# All tests pass ✅
# No regressions in existing tests ✅
# User can manually test strategy selection ✅
```

### Phase 2 Gate
```bash
pytest tests/strategy/test_plateau* tests/strategy/test_auto* -v
pytest tests/agents/test_adversary_plateau.py -v
pytest tests/integration/test_phase2_integration.py -v
# All tests pass ✅
# Plateau detection works in manual test ✅
# No regressions ✅
```

### Phase 3 Gate
```bash
pytest tests/hyperparameter/ -v
pytest tests/trust/test_hyperparameter_validation.py -v
pytest tests/integration/test_phase3_integration.py -v
# All tests pass ✅
# CMA-ES optimizes simple function ✅
# Santa sparrow test shows improvement ✅
# No regressions ✅
```

### Phase 4 Gate
```bash
pytest tests/ -v --cov=evolve_sdk
# All tests pass ✅
# Coverage >80% ✅
# End-to-end santa test works ✅
# Performance overhead <10% ✅
# Documentation complete ✅
```

---

## Quick Reference: File Creation Order

Copy this checklist for tracking progress:

```
PHASE 1:
□ evolve_sdk/strategy/__init__.py
□ evolve_sdk/strategy/types.py
□ evolve_sdk/ui/__init__.py
□ evolve_sdk/ui/user_interface.py
□ evolve_sdk/strategy/problem_analyzer.py
□ evolve_sdk/strategy/strategy_selector.py
□ evolve_sdk/memory/queries.py (modify)
□ evolve_sdk/memory/schemas.py (modify)
□ evolve_sdk/runner.py (modify)

PHASE 2:
□ evolve_sdk/strategy/types.py (extend)
□ evolve_sdk/strategy/plateau_detector.py
□ evolve_sdk/agents/adversary.py (extend)
□ evolve_sdk/strategy/auto_switcher.py
□ evolve_sdk/memory/schemas.py (extend)
□ evolve_sdk/runner.py (extend)

PHASE 3:
□ evolve_sdk/hyperparameter/__init__.py
□ evolve_sdk/hyperparameter/types.py
□ evolve_sdk/hyperparameter/search_methods.py
□ evolve_sdk/hyperparameter/detector.py
□ evolve_sdk/agents/adversary.py (extend)
□ evolve_sdk/hyperparameter/evolver.py
□ evolve_sdk/trust/validators.py (extend)
□ evolve_sdk/memory/schemas.py (extend)
□ evolve_sdk/runner.py (extend)

PHASE 4:
□ evolve_sdk/orchestrator.py
□ evolve_sdk/ui/enhanced_ui.py
□ evolve_sdk/recovery.py
□ docs/QUICKSTART.md
□ docs/STRATEGIES.md
□ docs/HYPERPARAMETERS.md
□ docs/TROUBLESHOOTING.md
□ examples/simple_optimization.py
□ examples/santa_hyperparameter.py
□ examples/advanced_strategies.py
□ README.md (update)
```

---

## Notes for Context Clears

When resuming after context clear:

1. **Check current phase**: Read `roadmap/PHASE[N]_*.md`
2. **Check progress**: Look at file checklist above
3. **Run tests**: Verify what's already working
4. **Continue from last incomplete file**

Each file is ~100-400 lines. Each test file is ~100-250 lines. Manageable incremental progress.
