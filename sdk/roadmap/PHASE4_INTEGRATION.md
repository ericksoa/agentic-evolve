# Phase 4: Integration & User Experience

**Status**: 🔴 Not Started
**Priority**: P1
**Estimated Duration**: 3 days
**Dependencies**: Phase 1, Phase 2, Phase 3

## Overview

Bring everything together into a seamless, intelligent end-to-end experience. This phase focuses on:
- Unified orchestration of all strategies
- Polished user interface with clear communication
- Comprehensive error handling and recovery
- Documentation and examples
- Performance optimization
- Final validation on real-world problems

## Goals

- ✅ Seamless orchestration of all strategies
- ✅ Clear, helpful user communication throughout
- ✅ Robust error handling and graceful degradation
- ✅ Comprehensive documentation and examples
- ✅ Performance optimization
- ✅ Validated on multiple showcase projects

## Architecture Components

### 1. Evolution Orchestrator
**File**: `evolve_sdk/orchestrator.py`

Main coordinator that uses ALL SDK systems:
- Problem analyzer (Phase 1)
- Strategy selector (Phase 1)
- Plateau detector (Phase 2)
- Auto switcher (Phase 2)
- Hyperparameter detector/evolver (Phase 3)
- Memory, Adversary, Trust, UI

**Key Method**:
```python
def run(self, user_prompt: str, resume: bool = False) -> EvolutionResult:
    """
    Complete evolution flow:

    1. Analyze problem (Phase 1)
    2. Select initial strategy (Phase 1)
    3. For each generation:
       a. Check for plateau (Phase 2)
       b. Auto-switch if needed (Phase 2)
       c. Execute current strategy (Phase 3 if hyperparameter)
       d. Validate with trust system
       e. Checkpoint to memory
    4. Return results
    """
```

### 2. Enhanced User Interface
**File**: `evolve_sdk/ui/enhanced_ui.py`

Rich user interaction:
- Progress indicators with estimates
- Strategy switch notifications
- Parameter evolution progress
- Final summary with insights

**Methods**:
```python
def show_evolution_start(strategy: str, context: ProblemContext)
def show_generation_progress(gen: int, fitness: float, improvement: float)
def show_strategy_switch(switch: StrategySwitch)
def show_hyperparameter_progress(trial: int, best_fitness: float)
def show_final_summary(result: EvolutionResult)
```

### 3. Configuration Manager
**File**: `evolve_sdk/config_manager.py`

Unified configuration with validation:
```python
class ConfigManager:
    def load_config(path: str) -> Config
    def validate_config(config: Config) -> List[str]  # Errors
    def merge_with_defaults(config: Config) -> Config
```

### 4. Error Recovery
**File**: `evolve_sdk/recovery.py`

Graceful handling of failures:
- Adversary unavailable → continue without validation, warn user
- Parameter detection fails → fall back to code evolution
- Optimization fails → return original solution
- Memory corrupted → warn user, continue without history

## Integration Points

### With All Previous Phases

**Phase 1 (Strategy Selection)**:
- Orchestrator uses analyzer and selector at start
- Error handling if selection fails

**Phase 2 (Plateau Detection)**:
- Orchestrator checks plateau each generation
- Handles strategy switches mid-evolution

**Phase 3 (Hyperparameter Evolution)**:
- Orchestrator executes hyperparameter strategy
- Shows progress during optimization

### With Existing SDK Systems

**Memory**:
- Orchestrator loads checkpoints on resume
- Saves checkpoints every N generations
- Stores complete evolution trace

**Adversary**:
- Validates plateaus, parameter extraction, improvements
- Graceful degradation if unavailable

**Trust**:
- Validates all improvements before accepting
- Rejects suspicious gains

**UI**:
- Clear communication at every decision point
- Progress indicators during long operations

## Files to Create

```
sdk/evolve_sdk/
├── orchestrator.py           # Main orchestration
├── config_manager.py         # Configuration handling
├── recovery.py               # Error recovery
└── ui/
    └── enhanced_ui.py        # Rich user interface

sdk/examples/
├── santa_hyperparameter.py   # Santa sparrow optimization
├── simple_optimization.py    # Basic usage
└── advanced_strategies.py    # All features demo

sdk/docs/
├── QUICKSTART.md            # Getting started guide
├── STRATEGIES.md            # Strategy selection guide
├── HYPERPARAMETERS.md       # Hyperparameter evolution guide
└── TROUBLESHOOTING.md       # Common issues

sdk/tests/integration/
├── test_end_to_end.py
├── test_error_recovery.py
└── test_performance.py
```

## Files to Modify

- `evolve_sdk/__main__.py` - Use orchestrator instead of direct evolution loop
- `evolve_sdk/evolution.py` - Extract logic into orchestrator
- `README.md` - Update with new capabilities
- `pyproject.toml` - Final dependency audit

## Validation Tests

### Test 1: End-to-End Automatic Strategy Selection
```python
def test_end_to_end_automatic_strategy_selection():
    """
    Full evolution with automatic strategy selection.

    Scenario:
    1. User asks to optimize algorithm
    2. System detects it has parameters
    3. Tries hyperparameter evolution first
    4. Gets stuck after 20 generations
    5. Auto-switches to code evolution
    6. Finds improvement
    """
    orchestrator = EvolutionOrchestrator(config, memory, ui, adversary, trust)

    initial_solution = Solution(code='''
class Solver:
    def __init__(self, lr=0.01, iters=100):
        self.lr = lr
        self.iters = iters
    def solve(self, x):
        result = x
        for _ in range(self.iters):
            result = result - self.lr * gradient(result)
        return result
''', fitness=0.5)

    result = orchestrator.run(
        user_prompt="optimize this solver",
        resume=False
    )

    # Verify strategy progression
    generations = memory.query(frame_type="generation")
    strategies = [g.metadata['strategy'] for g in generations]

    # Should start with hyperparameter_evolution (has params)
    assert strategies[0] == "hyperparameter_evolution"

    # Should switch when plateaued
    assert len(set(strategies)) > 1

    # Should improve
    assert result.champion.fitness > initial_solution.fitness
```

### Test 2: Error Recovery
```python
def test_graceful_degradation():
    """System recovers gracefully from component failures."""

    # Adversary unavailable
    orchestrator = EvolutionOrchestrator(
        config, memory, ui,
        adversary=None,  # Simulates failure
        trust=trust
    )

    # Should still work, just warn user
    result = orchestrator.run("optimize algorithm")
    assert result is not None
    assert "adversary unavailable" in ui.warnings_shown[0].lower()
```

### Test 3: Resume After Crash
```python
def test_resume_after_crash():
    """Evolution can resume from checkpoint after crash."""

    orchestrator = EvolutionOrchestrator(config, memory, ui, adversary, trust)

    # Start evolution
    orchestrator.run("optimize algorithm", resume=False)

    # Simulate crash at generation 15
    checkpoint = memory.load_checkpoint()
    assert checkpoint.generation == 15

    # Resume
    orchestrator2 = EvolutionOrchestrator(config, memory, ui, adversary, trust)
    result = orchestrator2.run("optimize algorithm", resume=True)

    # Should continue from generation 15
    assert result.generations_completed > 15
```

### Test 4: User Communication
```python
def test_user_communication():
    """User receives clear communication throughout."""
    ui = MockUserInterface()
    orchestrator = EvolutionOrchestrator(config, memory, ui, adversary, trust)

    result = orchestrator.run("optimize algorithm")

    # Check user was informed
    assert ui.messages_count > 0
    assert any("strategy" in msg.lower() for msg in ui.messages)

    # Check final summary shown
    assert ui.final_summary_shown == True
    assert result.champion in ui.final_summary_data
```

### Test 5: Performance
```python
def test_performance_overhead():
    """Orchestration overhead is minimal."""
    import time

    # Baseline: direct evolution
    start = time.time()
    direct_result = run_direct_evolution(max_gens=10)
    direct_time = time.time() - start

    # With orchestration
    start = time.time()
    orchestrator = EvolutionOrchestrator(config, memory, ui, adversary, trust)
    orchestrated_result = orchestrator.run("optimize", max_gens=10)
    orchestrated_time = time.time() - start

    # Overhead should be <10%
    overhead = (orchestrated_time - direct_time) / direct_time
    assert overhead < 0.10
```

### Test 6: Real-World Validation
```python
def test_santa_sparrow_optimization():
    """
    Real-world test: Optimize Santa sparrow algorithm.

    This replicates what Gen100 should have done.
    """
    # Load Gen100 sparrow implementation
    sparrow_code = load_file("showcase/santa-2025-packing/rust/src/sparrow.rs")
    solution = Solution(code=sparrow_code, fitness=None)

    # Configure for santa challenge
    config = Config(
        max_generations=50,
        plateau_patience=10,
        hyperparameter_budget=100
    )

    # Run orchestrator
    orchestrator = EvolutionOrchestrator(config, memory, ui, adversary, trust)
    result = orchestrator.run(
        "optimize this packing algorithm for santa challenge"
    )

    # Verify:
    # 1. Detected parameters in sparrow config
    params = memory.query(frame_type="parameter_detection")[-1]
    assert len(params.metadata['parameters']) >= 5

    # 2. Tried hyperparameter evolution
    strategies = [g.metadata['strategy'] for g in memory.query(frame_type="generation")]
    assert "hyperparameter_evolution" in strategies

    # 3. Found improvements (should beat Gen100 score of 12.33)
    assert result.champion.fitness < 12.0  # Lower is better for packing
```

## Success Criteria

- [x] Complete end-to-end evolution works seamlessly
- [x] All strategies properly orchestrated
- [x] User receives clear, helpful communication
- [x] Graceful error handling and recovery
- [x] Can resume from checkpoints
- [x] Performance overhead <10%
- [x] Validated on real-world problems (santa, others)
- [x] Documentation complete
- [x] All tests pass
- [x] No regressions

## Implementation Checklist

### Step 1: Orchestrator (Day 1)
- [ ] Create `orchestrator.py`
- [ ] Implement `run()` method with full flow
- [ ] Implement `_execute_strategy()` dispatch
- [ ] Add checkpoint save/load
- [ ] Write orchestrator tests

### Step 2: Enhanced UI (Day 1)
- [ ] Create `ui/enhanced_ui.py`
- [ ] Implement progress indicators
- [ ] Implement strategy switch notifications
- [ ] Implement final summary
- [ ] Test UI with mock data

### Step 3: Error Recovery (Day 2)
- [ ] Create `recovery.py`
- [ ] Implement graceful degradation for each component
- [ ] Add fallback strategies
- [ ] Test failure scenarios

### Step 4: Configuration (Day 2)
- [ ] Create `config_manager.py`
- [ ] Implement config validation
- [ ] Create default config template
- [ ] Document all config options

### Step 5: Integration (Day 2)
- [ ] Update `__main__.py` to use orchestrator
- [ ] Refactor `evolution.py` logic into orchestrator
- [ ] Ensure backward compatibility
- [ ] Write integration tests

### Step 6: Documentation (Day 3)
- [ ] Write QUICKSTART.md
- [ ] Write STRATEGIES.md
- [ ] Write HYPERPARAMETERS.md
- [ ] Write TROUBLESHOOTING.md
- [ ] Create examples
- [ ] Update main README

### Step 7: Validation (Day 3)
- [ ] Test on santa-2025-packing
- [ ] Test on other showcases
- [ ] Performance profiling
- [ ] Fix any issues found
- [ ] Final polish

## Configuration Schema

**File**: `evolve_config_schema.json`
```json
{
  "max_generations": 100,
  "checkpoint_interval": 10,

  "strategy_selection": {
    "enabled": true,
    "auto_select": false
  },

  "plateau_detection": {
    "enabled": true,
    "patience": 20,
    "min_improvement": 0.01,
    "min_diversity": 0.3,
    "require_adversary_validation": true,
    "auto_switch": true
  },

  "hyperparameter_evolution": {
    "enabled": true,
    "default_method": "cmaes",
    "default_budget": 100,
    "ask_user_confirmation": true
  },

  "recovery": {
    "continue_on_adversary_failure": true,
    "continue_on_trust_failure": false,
    "continue_on_memory_failure": true,
    "fallback_strategy": "code_evolution"
  },

  "ui": {
    "verbose": true,
    "show_progress": true,
    "show_strategy_switches": true,
    "show_final_summary": true
  }
}
```

## Example Usage

**File**: `examples/santa_hyperparameter.py`
```python
"""
Example: Optimize Santa 2025 sparrow algorithm hyperparameters.

This demonstrates what Gen100 should have done.
"""
from evolve_sdk import EvolutionOrchestrator, Config

# Load sparrow algorithm with default parameters
with open("showcase/santa-2025-packing/python/sparrow.py") as f:
    sparrow_code = f.read()

# Configure evolution
config = Config.from_file("evolve_config.json")
config.max_generations = 50
config.hyperparameter_evolution.budget = 200

# Create orchestrator
orchestrator = EvolutionOrchestrator(config)

# Run optimization
result = orchestrator.run(
    "Optimize sparrow algorithm parameters for santa packing challenge"
)

# Show results
print(f"Original fitness: {result.original_fitness:.2f}")
print(f"Improved fitness: {result.champion.fitness:.2f}")
print(f"Improvement: {result.improvement:.1%}")
print(f"\nBest parameters:")
for name, value in result.best_hyperparameters.items():
    print(f"  {name}: {value}")
```

## Documentation Outline

### QUICKSTART.md
```markdown
# Quick Start Guide

## Installation
## Basic Usage
## Your First Evolution Run
## Understanding the Output
## Next Steps
```

### STRATEGIES.md
```markdown
# Strategy Selection Guide

## Available Strategies
- Code Evolution
- Hyperparameter Evolution
- Alternative Paradigm
- Population-Based

## How Strategies Are Selected
## Manual Strategy Selection
## When to Use Each Strategy
## Strategy Switching
```

### HYPERPARAMETERS.md
```markdown
# Hyperparameter Evolution Guide

## What Are Hyperparameters?
## When to Use Hyperparameter Evolution
## How It Works
## Optimization Methods
## Configuring Budget
## Interpreting Results
## Advanced: Custom Parameter Spaces
```

### TROUBLESHOOTING.md
```markdown
# Troubleshooting Guide

## Common Issues
- No parameters detected
- Evolution not improving
- Strategy switch not happening
- Adversary validation failing

## Error Messages
## Performance Issues
## Getting Help
```

## Non-Goals (Out of Scope)

- ❌ Web UI or dashboard
- ❌ Distributed execution
- ❌ Cloud integration (separate roadmap)
- ❌ Multi-objective optimization

## Future Enhancements (Post-Phase 4)

- Interactive web dashboard
- Telemetry and analytics
- Cloud execution backends
- Transfer learning across problems
- Automated hyperparameter bound discovery

## Dependencies

**SDK Components**:
- Phase 1 - REQUIRED
- Phase 2 - REQUIRED
- Phase 3 - REQUIRED
- All existing SDK systems

**Python Packages**: None new

## Migration Notes

**Breaking Changes**:
- Main entry point changes from direct evolution loop to orchestrator
- Old usage still works with deprecation warning

**Migration Guide**:
```python
# Old way (still works but deprecated)
from evolve_sdk import run_evolution
result = run_evolution(prompt, config)

# New way
from evolve_sdk import EvolutionOrchestrator
orchestrator = EvolutionOrchestrator(config)
result = orchestrator.run(prompt)
```

## Rollout Plan

1. **Day 1**: Core orchestration implementation
2. **Day 2**: Error handling and configuration
3. **Day 3**: Documentation and validation
4. **Day 4**: Beta release to showcase projects
5. **Day 5**: Collect feedback
6. **Day 6**: Iterate and fix issues
7. **Day 7**: Stable release

## Performance Targets

- Orchestration overhead: <10% of total runtime
- Memory overhead: <50MB for orchestrator state
- Startup time: <500ms for configuration and setup
- Checkpoint save: <100ms

## Notes

- Focus on user experience - clear communication is critical
- Every decision point should be explained to user
- Graceful degradation better than hard failures
- Performance matters - don't slow down evolution loop
- Documentation as important as code
