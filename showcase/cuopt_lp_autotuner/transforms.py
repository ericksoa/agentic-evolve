"""
transforms.py - Safe formulation transforms for LP instances.

All transforms are:
1. Solution-preserving: the optimal solution can be recovered exactly
2. Invertible: we can map the transformed solution back to the original
3. Fast: O(nnz) or better complexity

Transforms available:
- Geometric row/column scaling (equilibration)
- Row reordering (by degree, coefficient range)
- Column reordering (by degree, bounds)
- RHS normalization
- Bound tightening (simple, provably safe cases only)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from scipy.sparse import csr_matrix
from enum import Enum


class TransformType(Enum):
    """Available transform types."""
    ROW_SCALE = "row_scale"
    COL_SCALE = "col_scale"
    ROW_REORDER = "row_reorder"
    COL_REORDER = "col_reorder"
    RHS_NORMALIZE = "rhs_normalize"
    BOUND_TIGHTEN = "bound_tighten"


@dataclass
class TransformResult:
    """Result of applying a transform, with inverse information."""
    transform_type: TransformType
    # Scaling factors or permutation indices for inversion
    inverse_data: Dict[str, Any]
    # Whether transform had any effect
    had_effect: bool


@dataclass
class LPInstance:
    """
    LP instance in canonical form.

    Minimize: c' x
    Subject to: A x {<=, =, >=} b (based on sense)
    Bounds: lb <= x <= ub

    Stored in CSR format for A.
    """
    c: np.ndarray           # Objective (n_vars,)
    A_data: np.ndarray      # Matrix values (nnz,)
    A_indices: np.ndarray   # Column indices (nnz,)
    A_indptr: np.ndarray    # Row pointers (n_cons+1,)
    b: np.ndarray           # RHS (n_cons,)
    lb: np.ndarray          # Lower bounds (n_vars,)
    ub: np.ndarray          # Upper bounds (n_vars,)
    sense: np.ndarray       # Constraint senses (n_cons,): -1=<=, 0==, 1=>=
    name: str = "unnamed"

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_cons(self) -> int:
        return len(self.b)

    @property
    def nnz(self) -> int:
        return len(self.A_data)

    def copy(self) -> 'LPInstance':
        """Create a deep copy."""
        return LPInstance(
            c=self.c.copy(),
            A_data=self.A_data.copy(),
            A_indices=self.A_indices.copy(),
            A_indptr=self.A_indptr.copy(),
            b=self.b.copy(),
            lb=self.lb.copy(),
            ub=self.ub.copy(),
            sense=self.sense.copy(),
            name=self.name,
        )

    def to_csr(self) -> csr_matrix:
        """Convert to scipy CSR matrix."""
        return csr_matrix(
            (self.A_data, self.A_indices, self.A_indptr),
            shape=(self.n_cons, self.n_vars),
        )


class TransformPipeline:
    """
    Manages a sequence of transforms and their inverses.

    Usage:
        pipeline = TransformPipeline()
        lp_transformed = pipeline.apply(lp, [
            (TransformType.ROW_SCALE, {}),
            (TransformType.COL_SCALE, {}),
        ])
        # After solving, recover original solution:
        x_original = pipeline.inverse_solution(x_transformed)
    """

    def __init__(self):
        self.transforms: List[TransformResult] = []

    def apply(
        self,
        lp: LPInstance,
        transform_specs: List[Tuple[TransformType, Dict[str, Any]]],
    ) -> LPInstance:
        """
        Apply a sequence of transforms to an LP.

        Args:
            lp: Input LP instance
            transform_specs: List of (TransformType, params) tuples

        Returns:
            Transformed LP instance
        """
        self.transforms = []
        result = lp.copy()

        for transform_type, params in transform_specs:
            if transform_type == TransformType.ROW_SCALE:
                result, tr = apply_row_scaling(result, **params)
            elif transform_type == TransformType.COL_SCALE:
                result, tr = apply_col_scaling(result, **params)
            elif transform_type == TransformType.ROW_REORDER:
                result, tr = apply_row_reorder(result, **params)
            elif transform_type == TransformType.COL_REORDER:
                result, tr = apply_col_reorder(result, **params)
            elif transform_type == TransformType.RHS_NORMALIZE:
                result, tr = apply_rhs_normalize(result, **params)
            elif transform_type == TransformType.BOUND_TIGHTEN:
                result, tr = apply_bound_tighten(result, **params)
            else:
                raise ValueError(f"Unknown transform: {transform_type}")

            self.transforms.append(tr)

        return result

    def inverse_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Map a solution from transformed space back to original.

        Args:
            x: Solution in transformed space

        Returns:
            Solution in original space
        """
        result = x.copy()

        # Apply inverses in reverse order
        for tr in reversed(self.transforms):
            if not tr.had_effect:
                continue

            if tr.transform_type == TransformType.COL_SCALE:
                # x_orig = x_trans / col_scale
                result = result / tr.inverse_data['col_scale']

            elif tr.transform_type == TransformType.COL_REORDER:
                # Unpermute: x_orig[perm] = x_trans
                inv_perm = tr.inverse_data['inv_perm']
                result = result[inv_perm]

            # Row transforms don't affect x
            # RHS normalize doesn't affect x
            # Bound tightening doesn't change solution mapping

        return result

    def inverse_objective(self, obj_transformed: float) -> float:
        """
        Map objective value from transformed space back to original.

        For scaling transforms, the objective may need adjustment.
        """
        result = obj_transformed

        for tr in reversed(self.transforms):
            if not tr.had_effect:
                continue

            if tr.transform_type == TransformType.RHS_NORMALIZE:
                # Objective was scaled
                result = result * tr.inverse_data.get('obj_scale', 1.0)

        return result


def apply_row_scaling(
    lp: LPInstance,
    method: str = 'geometric',
) -> Tuple[LPInstance, TransformResult]:
    """
    Apply row scaling to equilibrate constraint matrix.

    Scales each row so that max |a_ij| in row i is approximately 1.

    This is solution-preserving as it just divides each constraint by a constant.

    Args:
        lp: Input LP
        method: 'geometric' (default) or 'arithmetic'

    Returns:
        (scaled_lp, transform_result)
    """
    result = lp.copy()

    # Compute row scales
    n_cons = lp.n_cons
    row_scale = np.ones(n_cons)

    for i in range(n_cons):
        start, end = lp.A_indptr[i], lp.A_indptr[i + 1]
        if start < end:
            row_vals = np.abs(lp.A_data[start:end])
            max_val = np.max(row_vals)
            if max_val > 1e-15:
                row_scale[i] = 1.0 / max_val

    # Apply scaling
    for i in range(n_cons):
        start, end = lp.A_indptr[i], lp.A_indptr[i + 1]
        result.A_data[start:end] = lp.A_data[start:end] * row_scale[i]
        result.b[i] = lp.b[i] * row_scale[i]

    had_effect = not np.allclose(row_scale, 1.0)

    return result, TransformResult(
        transform_type=TransformType.ROW_SCALE,
        inverse_data={'row_scale': row_scale},
        had_effect=had_effect,
    )


def apply_col_scaling(
    lp: LPInstance,
    method: str = 'geometric',
) -> Tuple[LPInstance, TransformResult]:
    """
    Apply column scaling to equilibrate variables.

    Scales each column so that max |a_ij| in column j is approximately 1.
    Also scales the objective and bounds accordingly.

    Transform: x_new = x_old * col_scale
    Inverse: x_old = x_new / col_scale

    Args:
        lp: Input LP
        method: 'geometric' (default)

    Returns:
        (scaled_lp, transform_result)
    """
    result = lp.copy()
    n_vars = lp.n_vars

    # Compute column max magnitudes
    col_max = np.zeros(n_vars)
    for i in range(lp.n_cons):
        start, end = lp.A_indptr[i], lp.A_indptr[i + 1]
        for k in range(start, end):
            j = lp.A_indices[k]
            col_max[j] = max(col_max[j], abs(lp.A_data[k]))

    # Compute column scales (avoid division by zero)
    col_scale = np.ones(n_vars)
    mask = col_max > 1e-15
    col_scale[mask] = 1.0 / col_max[mask]

    # Apply scaling to A
    for i in range(lp.n_cons):
        start, end = lp.A_indptr[i], lp.A_indptr[i + 1]
        for k in range(start, end):
            j = lp.A_indices[k]
            result.A_data[k] = lp.A_data[k] * col_scale[j]

    # Scale objective: c_new = c_old / col_scale
    result.c = lp.c / col_scale

    # Scale bounds: if x_new = x_old * col_scale, then
    # lb_old <= x_old <= ub_old becomes
    # lb_old * col_scale <= x_new <= ub_old * col_scale
    result.lb = lp.lb * col_scale
    result.ub = lp.ub * col_scale

    had_effect = not np.allclose(col_scale, 1.0)

    return result, TransformResult(
        transform_type=TransformType.COL_SCALE,
        inverse_data={'col_scale': col_scale},
        had_effect=had_effect,
    )


def apply_row_reorder(
    lp: LPInstance,
    method: str = 'degree_ascending',
) -> Tuple[LPInstance, TransformResult]:
    """
    Reorder rows (constraints) for better cache locality.

    Methods:
    - 'degree_ascending': sparse rows first
    - 'degree_descending': dense rows first
    - 'rhs_ascending': small RHS first

    This is solution-preserving (just reorders constraints).

    Args:
        lp: Input LP
        method: Reordering method

    Returns:
        (reordered_lp, transform_result)
    """
    n_cons = lp.n_cons

    # Compute row degrees
    row_degrees = np.diff(lp.A_indptr)

    # Get permutation
    if method == 'degree_ascending':
        perm = np.argsort(row_degrees)
    elif method == 'degree_descending':
        perm = np.argsort(-row_degrees)
    elif method == 'rhs_ascending':
        perm = np.argsort(np.abs(lp.b))
    else:
        perm = np.arange(n_cons)

    # Check if permutation is identity
    if np.array_equal(perm, np.arange(n_cons)):
        return lp, TransformResult(
            transform_type=TransformType.ROW_REORDER,
            inverse_data={'perm': perm, 'inv_perm': perm},
            had_effect=False,
        )

    # Apply permutation
    A_csr = lp.to_csr()
    A_perm = A_csr[perm]

    result = LPInstance(
        c=lp.c.copy(),
        A_data=A_perm.data.copy(),
        A_indices=A_perm.indices.copy(),
        A_indptr=A_perm.indptr.copy(),
        b=lp.b[perm],
        lb=lp.lb.copy(),
        ub=lp.ub.copy(),
        sense=lp.sense[perm],
        name=lp.name,
    )

    inv_perm = np.argsort(perm)

    return result, TransformResult(
        transform_type=TransformType.ROW_REORDER,
        inverse_data={'perm': perm, 'inv_perm': inv_perm},
        had_effect=True,
    )


def apply_col_reorder(
    lp: LPInstance,
    method: str = 'degree_descending',
) -> Tuple[LPInstance, TransformResult]:
    """
    Reorder columns (variables) for better numerical behavior.

    Methods:
    - 'degree_descending': frequently appearing variables first
    - 'degree_ascending': rarely appearing variables first
    - 'bounds_first': bounded variables before free variables

    Inverse: x_orig[perm] = x_trans

    Args:
        lp: Input LP
        method: Reordering method

    Returns:
        (reordered_lp, transform_result)
    """
    n_vars = lp.n_vars

    # Compute column degrees
    col_degrees = np.bincount(lp.A_indices, minlength=n_vars)

    # Get permutation
    if method == 'degree_descending':
        perm = np.argsort(-col_degrees)
    elif method == 'degree_ascending':
        perm = np.argsort(col_degrees)
    elif method == 'bounds_first':
        # Bounded variables first, then free
        is_bounded = np.isfinite(lp.lb) | np.isfinite(lp.ub)
        perm = np.argsort(~is_bounded)  # False (bounded) sorts before True (free)
    else:
        perm = np.arange(n_vars)

    # Check if permutation is identity
    if np.array_equal(perm, np.arange(n_vars)):
        return lp, TransformResult(
            transform_type=TransformType.COL_REORDER,
            inverse_data={'perm': perm, 'inv_perm': perm},
            had_effect=False,
        )

    # Apply permutation
    inv_perm = np.argsort(perm)

    # Reorder objective and bounds
    c_new = lp.c[perm]
    lb_new = lp.lb[perm]
    ub_new = lp.ub[perm]

    # Reorder column indices in A
    A_indices_new = inv_perm[lp.A_indices]

    result = LPInstance(
        c=c_new,
        A_data=lp.A_data.copy(),
        A_indices=A_indices_new,
        A_indptr=lp.A_indptr.copy(),
        b=lp.b.copy(),
        lb=lb_new,
        ub=ub_new,
        sense=lp.sense.copy(),
        name=lp.name,
    )

    return result, TransformResult(
        transform_type=TransformType.COL_REORDER,
        inverse_data={'perm': perm, 'inv_perm': inv_perm},
        had_effect=True,
    )


def apply_rhs_normalize(
    lp: LPInstance,
    target_max: float = 1.0,
) -> Tuple[LPInstance, TransformResult]:
    """
    Normalize RHS values to a target range.

    Scales the entire problem so max |b_i| = target_max.
    This affects the objective value but not the optimal solution.

    Args:
        lp: Input LP
        target_max: Target maximum RHS magnitude

    Returns:
        (normalized_lp, transform_result)
    """
    rhs_max = np.max(np.abs(lp.b))

    if rhs_max < 1e-15:
        return lp, TransformResult(
            transform_type=TransformType.RHS_NORMALIZE,
            inverse_data={'scale': 1.0, 'obj_scale': 1.0},
            had_effect=False,
        )

    scale = target_max / rhs_max

    if abs(scale - 1.0) < 1e-10:
        return lp, TransformResult(
            transform_type=TransformType.RHS_NORMALIZE,
            inverse_data={'scale': 1.0, 'obj_scale': 1.0},
            had_effect=False,
        )

    result = lp.copy()
    result.A_data = lp.A_data * scale
    result.b = lp.b * scale

    # Scale bounds too
    result.lb = lp.lb * scale
    result.ub = lp.ub * scale

    return result, TransformResult(
        transform_type=TransformType.RHS_NORMALIZE,
        inverse_data={'scale': scale, 'obj_scale': scale},
        had_effect=True,
    )


def apply_bound_tighten(
    lp: LPInstance,
    tolerance: float = 1e-6,
) -> Tuple[LPInstance, TransformResult]:
    """
    Tighten variable bounds using simple constraint propagation.

    Only applies SAFE tightening:
    - Single-variable constraints that directly bound a variable
    - Does NOT do iterative propagation (could cause issues)

    This is solution-preserving because we only tighten, never loosen.

    Args:
        lp: Input LP
        tolerance: Numerical tolerance for bound improvements

    Returns:
        (tightened_lp, transform_result)
    """
    result = lp.copy()
    lb_new = result.lb.copy()
    ub_new = result.ub.copy()

    # Look for single-variable rows
    row_degrees = np.diff(lp.A_indptr)
    single_var_rows = np.where(row_degrees == 1)[0]

    for i in single_var_rows:
        start = lp.A_indptr[i]
        j = lp.A_indices[start]
        a_ij = lp.A_data[start]
        b_i = lp.b[i]
        sense_i = lp.sense[i]

        if abs(a_ij) < 1e-15:
            continue

        bound = b_i / a_ij

        # a_ij * x_j sense b_i
        if a_ij > 0:
            if sense_i == -1:  # x_j <= bound
                if np.isfinite(bound) and (not np.isfinite(ub_new[j]) or bound < ub_new[j] - tolerance):
                    ub_new[j] = min(ub_new[j], bound) if np.isfinite(ub_new[j]) else bound
            elif sense_i == 1:  # x_j >= bound
                if np.isfinite(bound) and (not np.isfinite(lb_new[j]) or bound > lb_new[j] + tolerance):
                    lb_new[j] = max(lb_new[j], bound) if np.isfinite(lb_new[j]) else bound
        else:  # a_ij < 0
            if sense_i == -1:  # x_j >= bound (inequality flips)
                if np.isfinite(bound) and (not np.isfinite(lb_new[j]) or bound > lb_new[j] + tolerance):
                    lb_new[j] = max(lb_new[j], bound) if np.isfinite(lb_new[j]) else bound
            elif sense_i == 1:  # x_j <= bound
                if np.isfinite(bound) and (not np.isfinite(ub_new[j]) or bound < ub_new[j] - tolerance):
                    ub_new[j] = min(ub_new[j], bound) if np.isfinite(ub_new[j]) else bound

    result.lb = lb_new
    result.ub = ub_new

    had_effect = not (np.allclose(lb_new, lp.lb, equal_nan=True) and
                      np.allclose(ub_new, lp.ub, equal_nan=True))

    return result, TransformResult(
        transform_type=TransformType.BOUND_TIGHTEN,
        inverse_data={'orig_lb': lp.lb, 'orig_ub': lp.ub},
        had_effect=had_effect,
    )


# Convenience function to get all safe transforms
def get_safe_transform_specs(
    enable_row_scale: bool = True,
    enable_col_scale: bool = True,
    enable_row_reorder: bool = True,
    enable_col_reorder: bool = True,
    enable_rhs_normalize: bool = False,
    enable_bound_tighten: bool = True,
    row_reorder_method: str = 'degree_ascending',
    col_reorder_method: str = 'degree_descending',
) -> List[Tuple[TransformType, Dict[str, Any]]]:
    """
    Build a list of transform specs from boolean toggles.

    This is the main entry point for the policy to specify transforms.
    """
    specs = []

    if enable_row_scale:
        specs.append((TransformType.ROW_SCALE, {'method': 'geometric'}))

    if enable_col_scale:
        specs.append((TransformType.COL_SCALE, {'method': 'geometric'}))

    if enable_row_reorder:
        specs.append((TransformType.ROW_REORDER, {'method': row_reorder_method}))

    if enable_col_reorder:
        specs.append((TransformType.COL_REORDER, {'method': col_reorder_method}))

    if enable_rhs_normalize:
        specs.append((TransformType.RHS_NORMALIZE, {'target_max': 1.0}))

    if enable_bound_tighten:
        specs.append((TransformType.BOUND_TIGHTEN, {'tolerance': 1e-6}))

    return specs
