"""
feature_extract.py - Fast structural feature extraction from LP/MPS files.

Extracts features used by the policy to select transforms and solver parameters:
- Size metrics: n_vars, n_cons, nnz, density
- Degree statistics: row/col min/max/mean/std
- Coefficient statistics: magnitude min/max/mean/std, fraction zero/near-zero
- RHS/bounds statistics: magnitude, fraction bounded/free
- Constraint sense distribution: <=, >=, =

All features are designed to be fast to compute (O(nnz) or better).
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import json


@dataclass
class LPFeatures:
    """Structural features of an LP instance."""

    # Size metrics
    n_vars: int
    n_cons: int
    nnz: int
    density: float  # nnz / (n_vars * n_cons)

    # Row degree stats (constraints)
    row_degree_min: int
    row_degree_max: int
    row_degree_mean: float
    row_degree_std: float

    # Column degree stats (variables)
    col_degree_min: int
    col_degree_max: int
    col_degree_mean: float
    col_degree_std: float

    # Coefficient magnitude stats (non-zeros only)
    coef_abs_min: float
    coef_abs_max: float
    coef_abs_mean: float
    coef_abs_std: float
    coef_range_log10: float  # log10(max/min), measure of numerical range

    # RHS stats
    rhs_abs_max: float
    rhs_abs_mean: float
    rhs_frac_zero: float

    # Bounds stats
    frac_vars_bounded_both: float  # both lb and ub finite
    frac_vars_bounded_lower: float  # only lb finite
    frac_vars_bounded_upper: float  # only ub finite
    frac_vars_free: float  # neither bound finite

    # Constraint sense distribution
    frac_leq: float  # <= constraints
    frac_geq: float  # >= constraints
    frac_eq: float   # = constraints

    # Objective stats
    obj_abs_max: float
    obj_abs_mean: float
    obj_nnz: int
    obj_density: float

    # Derived complexity indicators
    size_bucket: str  # 'tiny', 'small', 'medium', 'large', 'huge'
    density_bucket: str  # 'sparse', 'medium', 'dense'
    numerical_difficulty: str  # 'easy', 'moderate', 'hard'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LPFeatures':
        return cls(**d)


def compute_size_bucket(n_vars: int, n_cons: int) -> str:
    """Categorize LP size for policy bucketing."""
    total = n_vars + n_cons
    if total < 100:
        return 'tiny'
    elif total < 1000:
        return 'small'
    elif total < 10000:
        return 'medium'
    elif total < 100000:
        return 'large'
    else:
        return 'huge'


def compute_density_bucket(density: float) -> str:
    """Categorize matrix density."""
    if density < 0.01:
        return 'sparse'
    elif density < 0.1:
        return 'medium'
    else:
        return 'dense'


def compute_numerical_difficulty(coef_range_log10: float, rhs_abs_max: float, coef_abs_max: float) -> str:
    """Estimate numerical difficulty from coefficient ranges."""
    # High range indicates potential numerical issues
    if coef_range_log10 > 6 or (rhs_abs_max > 1e10 and coef_abs_max > 1e6):
        return 'hard'
    elif coef_range_log10 > 3:
        return 'moderate'
    else:
        return 'easy'


def extract_features(
    c: np.ndarray,           # Objective coefficients (n_vars,)
    A_data: np.ndarray,      # Constraint matrix values (nnz,)
    A_indices: np.ndarray,   # Constraint matrix column indices (nnz,)
    A_indptr: np.ndarray,    # Constraint matrix row pointers (n_cons+1,)
    b: np.ndarray,           # RHS values (n_cons,)
    lb: np.ndarray,          # Variable lower bounds (n_vars,)
    ub: np.ndarray,          # Variable upper bounds (n_vars,)
    sense: np.ndarray,       # Constraint senses: -1 (<=), 0 (=), 1 (>=)
) -> LPFeatures:
    """
    Extract structural features from LP in sparse CSR format.

    LP formulation: min c'x s.t. A*x {<=,=,>=} b, lb <= x <= ub

    Args:
        c: Objective coefficients
        A_data, A_indices, A_indptr: Constraint matrix in CSR format
        b: Right-hand side values
        lb, ub: Variable bounds
        sense: Constraint senses (-1=leq, 0=eq, 1=geq)

    Returns:
        LPFeatures dataclass with all extracted features
    """
    n_vars = len(c)
    n_cons = len(b)
    nnz = len(A_data)

    # Density
    density = nnz / (n_vars * n_cons) if n_vars > 0 and n_cons > 0 else 0.0

    # Row degrees (from CSR indptr)
    row_degrees = np.diff(A_indptr)
    row_degree_min = int(np.min(row_degrees)) if n_cons > 0 else 0
    row_degree_max = int(np.max(row_degrees)) if n_cons > 0 else 0
    row_degree_mean = float(np.mean(row_degrees)) if n_cons > 0 else 0.0
    row_degree_std = float(np.std(row_degrees)) if n_cons > 0 else 0.0

    # Column degrees (need to count)
    if nnz > 0:
        col_degrees = np.bincount(A_indices, minlength=n_vars)
        col_degree_min = int(np.min(col_degrees))
        col_degree_max = int(np.max(col_degrees))
        col_degree_mean = float(np.mean(col_degrees))
        col_degree_std = float(np.std(col_degrees))
    else:
        col_degree_min = col_degree_max = 0
        col_degree_mean = col_degree_std = 0.0

    # Coefficient magnitude stats
    if nnz > 0:
        abs_data = np.abs(A_data)
        nonzero_mask = abs_data > 1e-15
        if np.any(nonzero_mask):
            abs_nonzero = abs_data[nonzero_mask]
            coef_abs_min = float(np.min(abs_nonzero))
            coef_abs_max = float(np.max(abs_nonzero))
            coef_abs_mean = float(np.mean(abs_nonzero))
            coef_abs_std = float(np.std(abs_nonzero))
            coef_range_log10 = float(np.log10(coef_abs_max / coef_abs_min)) if coef_abs_min > 0 else 0.0
        else:
            coef_abs_min = coef_abs_max = coef_abs_mean = coef_abs_std = coef_range_log10 = 0.0
    else:
        coef_abs_min = coef_abs_max = coef_abs_mean = coef_abs_std = coef_range_log10 = 0.0

    # RHS stats
    if n_cons > 0:
        abs_b = np.abs(b)
        rhs_abs_max = float(np.max(abs_b))
        rhs_abs_mean = float(np.mean(abs_b))
        rhs_frac_zero = float(np.mean(abs_b < 1e-15))
    else:
        rhs_abs_max = rhs_abs_mean = 0.0
        rhs_frac_zero = 1.0

    # Bounds stats
    if n_vars > 0:
        lb_finite = np.isfinite(lb)
        ub_finite = np.isfinite(ub)
        frac_vars_bounded_both = float(np.mean(lb_finite & ub_finite))
        frac_vars_bounded_lower = float(np.mean(lb_finite & ~ub_finite))
        frac_vars_bounded_upper = float(np.mean(~lb_finite & ub_finite))
        frac_vars_free = float(np.mean(~lb_finite & ~ub_finite))
    else:
        frac_vars_bounded_both = frac_vars_bounded_lower = 0.0
        frac_vars_bounded_upper = frac_vars_free = 0.0

    # Constraint sense distribution
    if n_cons > 0:
        frac_leq = float(np.mean(sense == -1))
        frac_geq = float(np.mean(sense == 1))
        frac_eq = float(np.mean(sense == 0))
    else:
        frac_leq = frac_geq = frac_eq = 0.0

    # Objective stats
    if n_vars > 0:
        abs_c = np.abs(c)
        obj_abs_max = float(np.max(abs_c)) if n_vars > 0 else 0.0
        obj_abs_mean = float(np.mean(abs_c))
        obj_nnz = int(np.sum(abs_c > 1e-15))
        obj_density = obj_nnz / n_vars
    else:
        obj_abs_max = obj_abs_mean = obj_density = 0.0
        obj_nnz = 0

    # Derived buckets
    size_bucket = compute_size_bucket(n_vars, n_cons)
    density_bucket = compute_density_bucket(density)
    numerical_difficulty = compute_numerical_difficulty(coef_range_log10, rhs_abs_max, coef_abs_max)

    return LPFeatures(
        n_vars=n_vars,
        n_cons=n_cons,
        nnz=nnz,
        density=density,
        row_degree_min=row_degree_min,
        row_degree_max=row_degree_max,
        row_degree_mean=row_degree_mean,
        row_degree_std=row_degree_std,
        col_degree_min=col_degree_min,
        col_degree_max=col_degree_max,
        col_degree_mean=col_degree_mean,
        col_degree_std=col_degree_std,
        coef_abs_min=coef_abs_min,
        coef_abs_max=coef_abs_max,
        coef_abs_mean=coef_abs_mean,
        coef_abs_std=coef_abs_std,
        coef_range_log10=coef_range_log10,
        rhs_abs_max=rhs_abs_max,
        rhs_abs_mean=rhs_abs_mean,
        rhs_frac_zero=rhs_frac_zero,
        frac_vars_bounded_both=frac_vars_bounded_both,
        frac_vars_bounded_lower=frac_vars_bounded_lower,
        frac_vars_bounded_upper=frac_vars_bounded_upper,
        frac_vars_free=frac_vars_free,
        frac_leq=frac_leq,
        frac_geq=frac_geq,
        frac_eq=frac_eq,
        obj_abs_max=obj_abs_max,
        obj_abs_mean=obj_abs_mean,
        obj_nnz=obj_nnz,
        obj_density=obj_density,
        size_bucket=size_bucket,
        density_bucket=density_bucket,
        numerical_difficulty=numerical_difficulty,
    )


def extract_features_from_lp(lp: 'LPInstance') -> LPFeatures:
    """Extract features from an LPInstance object."""
    return extract_features(
        c=lp.c,
        A_data=lp.A_data,
        A_indices=lp.A_indices,
        A_indptr=lp.A_indptr,
        b=lp.b,
        lb=lp.lb,
        ub=lp.ub,
        sense=lp.sense,
    )


def quick_features(lp: 'LPInstance') -> Dict[str, Any]:
    """
    Extract minimal features for fast policy lookup.

    Returns only the features needed for bucketing decisions.
    """
    n_vars = len(lp.c)
    n_cons = len(lp.b)
    nnz = len(lp.A_data)
    density = nnz / (n_vars * n_cons) if n_vars > 0 and n_cons > 0 else 0.0

    return {
        'n_vars': n_vars,
        'n_cons': n_cons,
        'nnz': nnz,
        'density': density,
        'size_bucket': compute_size_bucket(n_vars, n_cons),
        'density_bucket': compute_density_bucket(density),
    }
