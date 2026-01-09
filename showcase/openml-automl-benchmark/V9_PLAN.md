# v9 Plan: Wrapper Mode (ThresholdOptimizer)

## Vision

A simple, composable wrapper that adds threshold optimization to ANY sklearn-compatible classifier.

```python
from adaptive_ensemble import ThresholdOptimizer

# Wrap any classifier
clf = ThresholdOptimizer(RandomForestClassifier(n_estimators=100))
clf.fit(X, y)
clf.set_operating_point(min_recall=0.95)
predictions = clf.predict(X_test)
```

## Design Principles

1. **Minimal API**: One class, obvious usage
2. **Non-invasive**: Doesn't modify wrapped estimator
3. **Full sklearn compatibility**: clone(), Pipeline, GridSearchCV all work
4. **Feature parity**: All v7/v8 features available (explain, plot, operating_points)

## Architecture

### Option 1: Separate Class (Recommended)

```
adaptive_ensemble/
├── threshold_classifier.py    # Existing ThresholdOptimizedClassifier
├── threshold_optimizer.py     # NEW: ThresholdOptimizer wrapper
├── _threshold_mixin.py        # NEW: Shared threshold logic
└── meta_learning/             # Existing
```

**Pros**: Clean separation, simpler new class, no risk to existing code
**Cons**: Some code duplication (mitigated by mixin)

### Option 2: Inheritance

```python
class ThresholdOptimizer(ThresholdOptimizedClassifier):
    """Simplified wrapper that uses provided estimator."""
```

**Pros**: Maximum code reuse
**Cons**: Inherits complexity, harder to simplify API

### Decision: Option 1 (Separate Class with Shared Mixin)

## Class Design

### ThresholdOptimizer

```python
class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds threshold optimization to any binary classifier.

    Parameters
    ----------
    estimator : estimator object
        A classifier with predict_proba() method. Will be cloned during fit.

    optimize_for : str, default='f1'
        Metric to optimize: 'f1', 'f2', 'precision', 'recall', 'balanced_accuracy'

    cv : int, default=5
        Number of cross-validation folds for threshold search.

    calibrate : bool or str, default=False
        Calibrate probabilities: False, True/'isotonic', 'sigmoid'

    strategy : str, default='auto'
        Detection strategy: 'auto', 'always', 'never'
        - 'auto': Use heuristics to decide if optimization will help
        - 'always': Always optimize threshold
        - 'never': Always use 0.5 (useful as baseline)

    safety_mode : bool, default=False
        Use holdout validation to detect overfitting.

    Attributes
    ----------
    estimator_ : estimator
        The fitted wrapped estimator.

    threshold_ : float
        The optimized threshold (or 0.5 if skipped).

    operating_points_ : dict
        Pareto frontier data (from v8).

    threshold_confidence_ : dict
        Confidence interval data (from v7).

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from adaptive_ensemble import ThresholdOptimizer
    >>>
    >>> # Wrap any classifier
    >>> clf = ThresholdOptimizer(RandomForestClassifier(n_estimators=100))
    >>> clf.fit(X_train, y_train)
    >>>
    >>> # Use v8 operating point selection
    >>> clf.set_operating_point(min_recall=0.95)
    >>> predictions = clf.predict(X_test)
    >>>
    >>> # Use v7 explainability
    >>> print(clf.explain())
    """

    def __init__(
        self,
        estimator,
        optimize_for='f1',
        cv=5,
        calibrate=False,
        strategy='auto',
        safety_mode=False,
        safety_margin=0.02,
    ):
        self.estimator = estimator
        self.optimize_for = optimize_for
        self.cv = cv
        self.calibrate = calibrate
        self.strategy = strategy
        self.safety_mode = safety_mode
        self.safety_margin = safety_margin

    def fit(self, X, y):
        """Fit wrapped estimator and optimize threshold."""
        # 1. Clone estimator (sklearn pattern)
        # 2. Fit with CV to get OOF probabilities
        # 3. Optimize threshold on OOF probabilities
        # 4. Compute operating points (v8)
        # 5. Compute confidence intervals (v7)
        # 6. Fit final estimator on all data
        pass

    def predict(self, X):
        """Predict using optimized threshold."""
        pass

    def predict_proba(self, X):
        """Return probability estimates from wrapped estimator."""
        pass

    # v7 features
    def explain(self) -> str:
        """Human-readable threshold analysis."""
        pass

    def plot(self, figsize=(10, 6), show=True):
        """Plot F1 vs threshold curve."""
        pass

    # v8 features
    def set_operating_point(self, **constraints):
        """Select operating point by business constraints."""
        pass

    def get_operating_point(self) -> dict:
        """Get current operating point details."""
        pass

    def list_operating_points(self, pareto_only=False):
        """List all operating points as DataFrame."""
        pass

    def plot_operating_points(self, figsize=(10, 8), show=True):
        """Visualize Pareto frontier."""
        pass
```

## Implementation Phases

### Phase 1: Core Wrapper (MVP)

**Goal**: Basic wrapper that optimizes threshold

**Files**:
- Create `adaptive_ensemble/threshold_optimizer.py`
- Update `adaptive_ensemble/__init__.py`

**Features**:
- `__init__` with estimator and basic params
- `fit()` - clone, CV, threshold optimization, final fit
- `predict()` - apply threshold
- `predict_proba()` - passthrough to estimator
- `get_params()` / `set_params()` - sklearn compatibility

**Tests**: 10 basic tests
- Wrapper works with LogisticRegression
- Wrapper works with RandomForest
- Wrapper works with XGBoost
- sklearn clone() works
- Pipeline integration works
- Threshold is optimized (not always 0.5)
- predict() uses threshold
- predict_proba() returns probabilities

### Phase 2: Strategy & Detection

**Goal**: Smart detection of when to optimize

**Features**:
- `strategy='auto'` - use heuristics
- `strategy='always'` - always optimize
- `strategy='never'` - baseline comparison
- Reuse detection logic from ThresholdOptimizedClassifier

**Tests**: 5 tests
- strategy='always' always optimizes
- strategy='never' always uses 0.5
- strategy='auto' sometimes skips

### Phase 3: v7 Features (Trust)

**Goal**: Port v7 features to wrapper

**Features**:
- `threshold_confidence_` attribute
- `explain()` method
- `plot()` method
- `safety_mode` parameter
- `calibrate` parameter

**Tests**: 8 tests
- Confidence intervals computed
- explain() returns string
- plot() creates figure
- safety_mode validates on holdout
- calibrate='isotonic' works
- calibrate='sigmoid' works

### Phase 4: v8 Features (Operating Points)

**Goal**: Port v8 Pareto frontier to wrapper

**Features**:
- `operating_points_` attribute
- `set_operating_point()` method
- `get_operating_point()` method
- `list_operating_points()` method
- `plot_operating_points()` method

**Tests**: 8 tests
- Operating points computed on fit
- set_operating_point(min_recall=X) works
- set_operating_point(min_precision=X) works
- get_operating_point() returns dict
- list_operating_points() returns DataFrame
- plot_operating_points() creates figure
- Operating point affects predict()

### Phase 5: Code Sharing

**Goal**: Extract shared logic to reduce duplication

**Approach**:
- Create `_threshold_mixin.py` with shared methods
- Refactor ThresholdOptimizedClassifier to use mixin
- ThresholdOptimizer uses same mixin

**Shared code**:
- `_optimize_threshold()` - threshold search logic
- `_compute_operating_points()` - Pareto calculation
- `_find_pareto_frontier()` - frontier detection
- `_compute_confidence_intervals()` - bootstrap CI
- `explain()` - explanation generation
- `plot()` / `plot_operating_points()` - visualization

### Phase 6: Documentation & Polish

**Goal**: Production-ready release

**Tasks**:
- Docstrings for all public methods
- Usage examples in docstrings
- Update RELEASE_NOTES.md
- Update README.md with wrapper examples
- Type hints throughout

## API Comparison

| Feature | ThresholdOptimizedClassifier | ThresholdOptimizer |
|---------|------------------------------|-------------------|
| Base model | Built-in (auto-select) | User-provided |
| Tuning | `tune_base_model=True` | User's responsibility |
| Feature scaling | `scale_features=True` | User's responsibility |
| Meta-detector | `use_meta_detector=True` | Not included (simpler) |
| Cost matrix | `cost_matrix={...}` | Included |
| All v7/v8 features | Yes | Yes |

## Migration Path

Users of `ThresholdOptimizedClassifier` can continue using it. For those who want the wrapper:

```python
# Before (v8)
clf = ThresholdOptimizedClassifier(
    base_estimator='xgboost',
    tune_base_model=True,
    optimize_for='f1',
)

# After (v9 wrapper) - if you want control over the base model
from xgboost import XGBClassifier
clf = ThresholdOptimizer(
    XGBClassifier(n_estimators=200, learning_rate=0.05),
    optimize_for='f1',
)
```

## Success Criteria

1. **Works with any classifier**: Tested with LogReg, RF, XGB, LightGBM, SVC
2. **Full sklearn compatibility**: clone, Pipeline, GridSearchCV, cross_val_score
3. **Feature parity**: All v7/v8 features work
4. **Performance**: No significant slowdown vs ThresholdOptimizedClassifier
5. **Test coverage**: 30+ new tests, all passing

## Estimated Scope

| Phase | New Tests | Cumulative |
|-------|-----------|------------|
| Phase 1 (Core) | 10 | 96 |
| Phase 2 (Strategy) | 5 | 101 |
| Phase 3 (v7) | 8 | 109 |
| Phase 4 (v8) | 8 | 117 |
| Phase 5 (Refactor) | 0 | 117 |
| Phase 6 (Polish) | 0 | 117 |

Total: ~31 new tests, targeting 117 total tests.

## Open Questions

1. **Should wrapper support multiclass?**
   - Recommendation: No, keep it binary-only for simplicity. Raise clear error.

2. **Should wrapper include meta-detector?**
   - Recommendation: No, it adds complexity. Use `strategy='auto'` with simpler heuristics.

3. **Should we deprecate ThresholdOptimizedClassifier?**
   - Recommendation: No, keep both. Different use cases.

4. **Name: ThresholdOptimizer vs ThresholdWrapper vs OptimalThreshold?**
   - Recommendation: `ThresholdOptimizer` - clear, action-oriented.
