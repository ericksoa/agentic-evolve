# v8 Plan: Operating Point Selection (Pareto Frontier)

## Vision

Transform ThresholdOptimizedClassifier from "find optimal threshold" to "explore the precision-recall tradeoff space and pick your operating point."

**Before (v7):**
```python
clf.fit(X, y)
print(f"Optimal threshold: {clf.optimal_threshold_}")  # 0.35
# User has no visibility into tradeoffs
```

**After (v8):**
```python
clf.fit(X, y)
clf.plot_operating_points()  # See full Pareto frontier

# Pick based on business needs
clf.set_operating_point(min_recall=0.95)      # "I can't miss positives"
clf.set_operating_point(min_precision=0.90)   # "I can't have false alarms"
clf.set_operating_point(max_fpr=0.05)         # "Max 5% false positive rate"

# See what you get
print(clf.get_operating_point())
# {'threshold': 0.28, 'precision': 0.72, 'recall': 0.95, 'f1': 0.82}
```

---

## Core Concepts

### Operating Point
A specific (threshold, precision, recall) tuple. Each threshold defines an operating point.

### Pareto Frontier
The set of non-dominated operating points. Point A dominates point B if A has both higher precision AND higher recall. The frontier contains points where you can't improve one metric without hurting the other.

```
Precision
    ^
1.0 |    *
    |     *
    |      *  <- Pareto frontier (these are the only points that matter)
    |       *
    |        *
0.5 |    x    *
    |  x   x   *
    |    x      *
    +-------------> Recall
              1.0
```

### Constraint-Based Selection
Instead of optimizing a single metric, users specify constraints:
- "Give me best precision where recall >= 95%"
- "Give me best recall where precision >= 90%"
- "Give me best F1 where FPR <= 5%"

---

## Implementation Plan

### Phase 1: Data Structure & Computation

#### 1.1 New Attribute: `operating_points_`

Store full metrics at each threshold tested:

```python
self.operating_points_ = {
    # Raw data (one entry per threshold)
    'thresholds': np.array([0.05, 0.10, 0.15, ..., 0.95]),
    'precisions': np.array([...]),
    'recalls': np.array([...]),
    'f1_scores': np.array([...]),
    'f2_scores': np.array([...]),
    'specificities': np.array([...]),  # TNR = TN / (TN + FP)
    'fprs': np.array([...]),           # FPR = FP / (FP + TN)
    'supports': np.array([...]),       # Number of positive predictions

    # Pareto frontier
    'pareto_mask': np.array([True, False, True, ...]),  # Which points are on frontier
    'n_pareto_points': 12,

    # Current selection
    'selected_index': 7,
    'selection_method': 'optimize_f1',  # or 'min_recall', 'min_precision', etc.
}
```

#### 1.2 New Method: `_compute_operating_points()`

```python
def _compute_operating_points(
    self,
    probs: np.ndarray,
    true_labels: np.ndarray,
    n_thresholds: int = 50,
) -> Dict:
    """
    Compute all metrics at multiple thresholds.

    Uses finer granularity than threshold optimization (50 vs 20 points)
    to give smoother Pareto frontier.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    results = {
        'thresholds': thresholds,
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'f2_scores': [],
        'specificities': [],
        'fprs': [],
        'supports': [],
    }

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        # Compute all metrics...

    # Find Pareto frontier
    results['pareto_mask'] = self._find_pareto_frontier(
        results['precisions'],
        results['recalls']
    )

    return results
```

#### 1.3 New Method: `_find_pareto_frontier()`

```python
def _find_pareto_frontier(
    self,
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> np.ndarray:
    """
    Find non-dominated points (Pareto frontier).

    Point i dominates point j if:
        precision[i] >= precision[j] AND recall[i] >= recall[j]
        AND at least one inequality is strict.

    Returns boolean mask of frontier points.
    """
    n = len(precisions)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if j dominates i
                if (precisions[j] >= precisions[i] and
                    recalls[j] >= recalls[i] and
                    (precisions[j] > precisions[i] or recalls[j] > recalls[i])):
                    is_pareto[i] = False
                    break

    return is_pareto
```

---

### Phase 2: Selection Methods

#### 2.1 New Method: `set_operating_point()`

```python
def set_operating_point(
    self,
    min_recall: float = None,
    min_precision: float = None,
    max_fpr: float = None,
    target_f1: float = None,
    target_f2: float = None,
    threshold: float = None,
) -> 'ThresholdOptimizedClassifier':
    """
    Set the operating point based on constraints.

    Only one constraint can be specified at a time.

    Parameters
    ----------
    min_recall : float, optional
        Find threshold achieving at least this recall, maximizing precision.
    min_precision : float, optional
        Find threshold achieving at least this precision, maximizing recall.
    max_fpr : float, optional
        Find threshold with FPR <= this value, maximizing recall.
    target_f1 : float, optional
        Find threshold closest to this F1 score.
    target_f2 : float, optional
        Find threshold closest to this F2 score.
    threshold : float, optional
        Directly set the threshold.

    Returns
    -------
    self : ThresholdOptimizedClassifier
        For method chaining.

    Examples
    --------
    >>> clf.fit(X, y)
    >>> clf.set_operating_point(min_recall=0.95)
    >>> print(clf.get_operating_point())
    {'threshold': 0.28, 'precision': 0.72, 'recall': 0.96, 'f1': 0.82}

    >>> clf.set_operating_point(min_precision=0.90)
    >>> print(clf.get_operating_point())
    {'threshold': 0.65, 'precision': 0.91, 'recall': 0.45, 'f1': 0.60}
    """
    if not hasattr(self, 'operating_points_'):
        raise RuntimeError("Model not fitted. Call fit() first.")

    # Count constraints
    constraints = [min_recall, min_precision, max_fpr, target_f1, target_f2, threshold]
    n_constraints = sum(c is not None for c in constraints)

    if n_constraints == 0:
        raise ValueError("Must specify at least one constraint.")
    if n_constraints > 1:
        raise ValueError("Only one constraint can be specified at a time.")

    ops = self.operating_points_

    if threshold is not None:
        # Direct threshold setting
        idx = np.argmin(np.abs(ops['thresholds'] - threshold))

    elif min_recall is not None:
        # Find highest precision where recall >= min_recall
        valid = ops['recalls'] >= min_recall
        if not valid.any():
            # No threshold achieves target, use lowest threshold (highest recall)
            idx = 0
            warnings.warn(f"No threshold achieves recall >= {min_recall}. "
                         f"Using lowest threshold (recall={ops['recalls'][0]:.2f})")
        else:
            # Among valid, find highest precision
            valid_indices = np.where(valid)[0]
            idx = valid_indices[np.argmax(ops['precisions'][valid])]

    elif min_precision is not None:
        # Find highest recall where precision >= min_precision
        valid = ops['precisions'] >= min_precision
        if not valid.any():
            idx = len(ops['thresholds']) - 1
            warnings.warn(f"No threshold achieves precision >= {min_precision}. "
                         f"Using highest threshold (precision={ops['precisions'][-1]:.2f})")
        else:
            valid_indices = np.where(valid)[0]
            idx = valid_indices[np.argmax(ops['recalls'][valid])]

    elif max_fpr is not None:
        # Find highest recall where FPR <= max_fpr
        valid = ops['fprs'] <= max_fpr
        if not valid.any():
            idx = len(ops['thresholds']) - 1
            warnings.warn(f"No threshold achieves FPR <= {max_fpr}.")
        else:
            valid_indices = np.where(valid)[0]
            idx = valid_indices[np.argmax(ops['recalls'][valid])]

    elif target_f1 is not None:
        idx = np.argmin(np.abs(ops['f1_scores'] - target_f1))

    elif target_f2 is not None:
        idx = np.argmin(np.abs(ops['f2_scores'] - target_f2))

    # Update selection
    self.operating_points_['selected_index'] = idx
    self.optimal_threshold_ = ops['thresholds'][idx]

    return self
```

#### 2.2 New Method: `get_operating_point()`

```python
def get_operating_point(self) -> Dict[str, float]:
    """
    Get the current operating point details.

    Returns
    -------
    point : dict
        - 'threshold': Current decision threshold
        - 'precision': Precision at this threshold
        - 'recall': Recall at this threshold
        - 'f1': F1 score at this threshold
        - 'f2': F2 score at this threshold
        - 'fpr': False positive rate
        - 'specificity': True negative rate
        - 'support': Number of positive predictions
        - 'is_pareto': Whether this point is on the Pareto frontier
    """
    if not hasattr(self, 'operating_points_'):
        raise RuntimeError("Model not fitted. Call fit() first.")

    ops = self.operating_points_
    idx = ops.get('selected_index', 0)

    return {
        'threshold': float(ops['thresholds'][idx]),
        'precision': float(ops['precisions'][idx]),
        'recall': float(ops['recalls'][idx]),
        'f1': float(ops['f1_scores'][idx]),
        'f2': float(ops['f2_scores'][idx]),
        'fpr': float(ops['fprs'][idx]),
        'specificity': float(ops['specificities'][idx]),
        'support': int(ops['supports'][idx]),
        'is_pareto': bool(ops['pareto_mask'][idx]),
    }
```

#### 2.3 New Method: `list_operating_points()`

```python
def list_operating_points(
    self,
    pareto_only: bool = False,
) -> 'pd.DataFrame':
    """
    Return all operating points as a DataFrame.

    Parameters
    ----------
    pareto_only : bool, default=False
        If True, only return points on the Pareto frontier.

    Returns
    -------
    df : pd.DataFrame
        Columns: threshold, precision, recall, f1, f2, fpr, specificity, is_pareto
    """
    import pandas as pd

    ops = self.operating_points_
    df = pd.DataFrame({
        'threshold': ops['thresholds'],
        'precision': ops['precisions'],
        'recall': ops['recalls'],
        'f1': ops['f1_scores'],
        'f2': ops['f2_scores'],
        'fpr': ops['fprs'],
        'specificity': ops['specificities'],
        'is_pareto': ops['pareto_mask'],
    })

    if pareto_only:
        df = df[df['is_pareto']].reset_index(drop=True)

    return df
```

---

### Phase 3: Visualization

#### 3.1 New Method: `plot_operating_points()`

```python
def plot_operating_points(
    self,
    x_metric: str = 'recall',
    y_metric: str = 'precision',
    show_pareto: bool = True,
    show_current: bool = True,
    show_iso_f1: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
) -> Optional['Figure']:
    """
    Plot the operating points with Pareto frontier.

    Parameters
    ----------
    x_metric : str, default='recall'
        Metric for x-axis. Options: 'recall', 'fpr', 'threshold'
    y_metric : str, default='precision'
        Metric for y-axis. Options: 'precision', 'specificity', 'f1'
    show_pareto : bool, default=True
        Highlight the Pareto frontier.
    show_current : bool, default=True
        Mark the current operating point.
    show_iso_f1 : bool, default=True
        Show iso-F1 curves (lines of constant F1).
    figsize : tuple, default=(10, 8)
        Figure size in inches.
    show : bool, default=True
        If True, display plot. If False, return Figure.

    Returns
    -------
    fig : Figure or None
        Figure object if show=False.
    """
    import matplotlib.pyplot as plt

    ops = self.operating_points_

    fig, ax = plt.subplots(figsize=figsize)

    # Get x and y data
    x_data = ops[f'{x_metric}s'] if x_metric != 'threshold' else ops['thresholds']
    y_data = ops[f'{y_metric}s'] if y_metric != 'threshold' else ops['thresholds']

    # Plot all points (non-Pareto as gray)
    non_pareto = ~ops['pareto_mask']
    ax.scatter(x_data[non_pareto], y_data[non_pareto],
               c='lightgray', s=30, alpha=0.5, label='Dominated')

    # Plot Pareto frontier
    if show_pareto:
        pareto = ops['pareto_mask']
        # Sort by x for line plot
        pareto_x = x_data[pareto]
        pareto_y = y_data[pareto]
        sort_idx = np.argsort(pareto_x)
        ax.plot(pareto_x[sort_idx], pareto_y[sort_idx],
                'b-', linewidth=2, label='Pareto Frontier')
        ax.scatter(pareto_x, pareto_y, c='blue', s=60, zorder=5)

    # Plot iso-F1 curves
    if show_iso_f1 and x_metric == 'recall' and y_metric == 'precision':
        for f1_val in [0.2, 0.4, 0.6, 0.8]:
            recall_range = np.linspace(0.01, 0.99, 100)
            # F1 = 2 * P * R / (P + R) => P = F1 * R / (2R - F1)
            precision_curve = f1_val * recall_range / (2 * recall_range - f1_val)
            valid = (precision_curve > 0) & (precision_curve <= 1)
            ax.plot(recall_range[valid], precision_curve[valid],
                    '--', color='gray', alpha=0.3, linewidth=1)
            # Label
            if valid.any():
                mid_idx = len(recall_range[valid]) // 2
                ax.annotate(f'F1={f1_val}',
                           (recall_range[valid][mid_idx], precision_curve[valid][mid_idx]),
                           fontsize=8, color='gray')

    # Mark current operating point
    if show_current:
        idx = ops.get('selected_index', 0)
        ax.scatter([x_data[idx]], [y_data[idx]],
                   c='red', s=200, marker='*', zorder=10,
                   label=f'Current (t={ops["thresholds"][idx]:.2f})')

        # Annotation with metrics
        point = self.get_operating_point()
        ax.annotate(
            f"P={point['precision']:.2f}\nR={point['recall']:.2f}\nF1={point['f1']:.2f}",
            (x_data[idx], y_data[idx]),
            xytext=(10, -20), textcoords='offset points',
            fontsize=9, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    # Labels
    ax.set_xlabel(x_metric.capitalize(), fontsize=12)
    ax.set_ylabel(y_metric.capitalize(), fontsize=12)
    ax.set_title('Operating Point Selection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig
```

---

### Phase 4: Integration with fit()

Modify `fit()` to compute operating points:

```python
def fit(self, X, y):
    # ... existing code ...

    # After threshold optimization, compute full operating points
    if self.n_classes_ == 2:
        self.operating_points_ = self._compute_operating_points(
            probs, true_labels, n_thresholds=50
        )
        # Set initial selection to the optimized threshold
        idx = np.argmin(np.abs(
            self.operating_points_['thresholds'] - self.optimal_threshold_
        ))
        self.operating_points_['selected_index'] = idx
        self.operating_points_['selection_method'] = f'optimize_{self.optimize_for}'

    # ... rest of existing code ...
```

---

## API Summary

### New Methods

| Method | Purpose |
|--------|---------|
| `set_operating_point(**constraints)` | Select threshold by constraint |
| `get_operating_point()` | Get current point details |
| `list_operating_points(pareto_only=False)` | Get all points as DataFrame |
| `plot_operating_points()` | Visualize Pareto frontier |

### Constraint Options for `set_operating_point()`

| Constraint | Behavior |
|------------|----------|
| `min_recall=0.95` | Best precision where recall >= 95% |
| `min_precision=0.90` | Best recall where precision >= 90% |
| `max_fpr=0.05` | Best recall where FPR <= 5% |
| `target_f1=0.80` | Closest to F1=0.80 |
| `target_f2=0.85` | Closest to F2=0.85 |
| `threshold=0.35` | Direct threshold setting |

---

## Test Plan

### Unit Tests

1. `test_operating_points_computed` - Check attribute exists after fit
2. `test_pareto_frontier_valid` - Verify no dominated points on frontier
3. `test_set_operating_point_min_recall` - Constraint satisfaction
4. `test_set_operating_point_min_precision` - Constraint satisfaction
5. `test_set_operating_point_max_fpr` - Constraint satisfaction
6. `test_set_operating_point_threshold` - Direct setting
7. `test_set_operating_point_unreachable` - Warning when constraint impossible
8. `test_get_operating_point` - Returns correct structure
9. `test_list_operating_points` - DataFrame output
10. `test_list_operating_points_pareto_only` - Filtering works
11. `test_plot_operating_points` - No errors, returns Figure
12. `test_operating_point_persists_predict` - Threshold used in predict()

### Integration Tests

1. `test_operating_point_in_pipeline` - Works with sklearn Pipeline
2. `test_operating_point_after_clone` - Preserved through clone

---

## Implementation Sequence

1. **Phase 1a**: Add `_compute_operating_points()` method
2. **Phase 1b**: Add `_find_pareto_frontier()` method
3. **Phase 1c**: Integrate into `fit()`, store `operating_points_`
4. **Phase 2a**: Add `set_operating_point()` method
5. **Phase 2b**: Add `get_operating_point()` method
6. **Phase 2c**: Add `list_operating_points()` method
7. **Phase 3**: Add `plot_operating_points()` method
8. **Phase 4**: Add all tests
9. **Phase 5**: Update documentation

---

## Success Criteria

1. **Functionality**: All constraint types work correctly
2. **Visualization**: Pareto frontier clearly visible
3. **Performance**: No significant slowdown (operating points computed from existing CV probs)
4. **Tests**: All 12+ tests passing
5. **Documentation**: Clear examples in NEXT_IMPROVEMENTS.md

---

## Future Extensions (v9+)

1. **Interactive plot**: Click to select operating point
2. **Multi-objective Pareto**: Include more than 2 metrics
3. **Cost curves**: Visualize expected cost at each operating point
4. **Threshold scheduling**: Different thresholds for different subgroups
