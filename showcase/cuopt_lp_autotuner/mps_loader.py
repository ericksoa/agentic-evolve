"""
mps_loader.py - Load LP problems from MPS/LP format files.

This module handles loading standard LP benchmark formats:
- MPS (Mathematical Programming System) format
- Free MPS format
- LP format (CPLEX-style)
- Compressed files (.gz, .bz2)

Uses scipy's MPS reader when available, with fallback to a custom parser.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmread
from typing import Tuple, Dict, Any, Optional, List
import gzip
import bz2
import os
from dataclasses import dataclass


@dataclass
class LPData:
    """
    Standard-form LP data structure.

    Represents: min c'x s.t. A_eq @ x = b_eq, lb <= x <= ub

    For constraints with inequality, we store original form separately.
    """
    # Objective
    c: np.ndarray           # (n_vars,) cost vector
    obj_offset: float = 0.0 # Constant in objective

    # Constraints in standard form (Ax = b with slacks)
    A_eq: Optional[csr_matrix] = None   # (m, n) constraint matrix
    b_eq: Optional[np.ndarray] = None   # (m,) RHS

    # Original constraints (before slack conversion)
    # sense: -1 = <=, 0 = =, 1 = >=
    A: Optional[csr_matrix] = None
    b: Optional[np.ndarray] = None
    sense: Optional[np.ndarray] = None

    # Bounds
    lb: Optional[np.ndarray] = None     # (n_vars,) lower bounds
    ub: Optional[np.ndarray] = None     # (n_vars,) upper bounds

    # Metadata
    name: str = ""
    var_names: Optional[List[str]] = None
    con_names: Optional[List[str]] = None

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_cons(self) -> int:
        if self.A is not None:
            return self.A.shape[0]
        if self.A_eq is not None:
            return self.A_eq.shape[0]
        return 0

    @property
    def nnz(self) -> int:
        if self.A is not None:
            return self.A.nnz
        if self.A_eq is not None:
            return self.A_eq.nnz
        return 0

    @property
    def density(self) -> float:
        if self.n_vars == 0 or self.n_cons == 0:
            return 0.0
        return self.nnz / (self.n_vars * self.n_cons)

    def to_cuopt_format(self) -> Dict[str, Any]:
        """
        Convert to cuOpt input format.

        cuOpt expects:
        - c: cost vector
        - A: constraint matrix (CSR)
        - b: RHS vector
        - sense: constraint sense array
        - lb, ub: variable bounds
        """
        return {
            "c": self.c,
            "A": self.A if self.A is not None else self.A_eq,
            "b": self.b if self.b is not None else self.b_eq,
            "sense": self.sense if self.sense is not None else np.zeros(self.n_cons),
            "lb": self.lb if self.lb is not None else np.zeros(self.n_vars),
            "ub": self.ub if self.ub is not None else np.full(self.n_vars, np.inf),
            "obj_offset": self.obj_offset,
            "name": self.name,
        }


def load_mps(filepath: str) -> LPData:
    """
    Load an LP problem from MPS file.

    Handles .mps, .mps.gz, .mps.bz2 extensions.
    Also handles Netlib compressed format.

    Args:
        filepath: Path to MPS file

    Returns:
        LPData structure with the LP
    """
    # Try scipy's internal loader first (handles Netlib format)
    try:
        result = _load_with_scipy(filepath)
        if result is not None:
            return result
    except Exception as e:
        pass

    # Try scipy's linprog with MPS file directly
    try:
        result = _load_with_scipy_linprog(filepath)
        if result is not None:
            return result
    except Exception as e:
        pass

    # Determine compression for custom parser
    if filepath.endswith('.gz'):
        opener = gzip.open
        mode = 'rt'
    elif filepath.endswith('.bz2'):
        opener = bz2.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'

    # Fall back to custom parser
    with opener(filepath, mode) as f:
        return _parse_mps(f, name=os.path.basename(filepath))


def _load_with_scipy_linprog(filepath: str) -> Optional[LPData]:
    """Try loading MPS by reading and parsing with HiGHS via linprog."""
    try:
        from scipy.optimize import linprog
        import tempfile
        import subprocess

        # HiGHS can read MPS files directly if we call it properly
        # For now, return None - this is a fallback path
        return None
    except Exception:
        return None


def _load_with_scipy(filepath: str) -> Optional[LPData]:
    """Try loading MPS using scipy's internal reader."""
    try:
        from scipy.optimize._linprog_highs import _load_mps

        # scipy's _load_mps returns (c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality, names)
        result = _load_mps(filepath)
        c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality, names = result

        n_vars = len(c)

        # Combine inequality and equality constraints
        # A_ub @ x <= b_ub -> sense = -1
        # A_eq @ x = b_eq -> sense = 0

        parts_A = []
        parts_b = []
        parts_sense = []

        if A_ub is not None and A_ub.shape[0] > 0:
            parts_A.append(csr_matrix(A_ub))
            parts_b.append(b_ub)
            parts_sense.append(np.full(A_ub.shape[0], -1))

        if A_eq is not None and A_eq.shape[0] > 0:
            parts_A.append(csr_matrix(A_eq))
            parts_b.append(b_eq)
            parts_sense.append(np.zeros(A_eq.shape[0]))

        if parts_A:
            from scipy.sparse import vstack
            A = vstack(parts_A)
            b = np.concatenate(parts_b)
            sense = np.concatenate(parts_sense).astype(int)
        else:
            A = csr_matrix((0, n_vars))
            b = np.array([])
            sense = np.array([])

        # Extract bounds
        lb = np.array([bd[0] if bd[0] is not None else -np.inf for bd in bounds])
        ub = np.array([bd[1] if bd[1] is not None else np.inf for bd in bounds])

        return LPData(
            c=c,
            A=A,
            b=b,
            sense=sense,
            lb=lb,
            ub=ub,
            name=os.path.basename(filepath),
        )

    except Exception as e:
        # scipy internal API may have changed
        return None


def _parse_mps(f, name: str = "") -> LPData:
    """
    Parse MPS format manually.

    MPS format sections:
    - NAME
    - OBJSENSE (optional, MIN or MAX)
    - ROWS (constraint names and types)
    - COLUMNS (variable coefficients)
    - RHS (right-hand sides)
    - RANGES (optional)
    - BOUNDS (optional)
    - ENDATA
    """
    lines = f.readlines()

    # Parse state
    section = None
    obj_name = None
    obj_sense = 1  # 1 = minimize, -1 = maximize

    row_names = []
    row_types = []  # 'N', 'L', 'G', 'E'
    col_names = []
    col_name_to_idx = {}

    # Sparse matrix data
    row_indices = []
    col_indices = []
    values = []

    rhs = {}  # row_name -> value
    bounds_lb = {}  # col_name -> lower bound
    bounds_ub = {}  # col_name -> upper bound

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        # Section headers
        upper = line.upper().strip()
        if upper == 'NAME' or upper.startswith('NAME '):
            section = 'NAME'
            if ' ' in upper:
                name = line.split(None, 1)[1].strip()
            continue
        elif upper == 'OBJSENSE':
            section = 'OBJSENSE'
            continue
        elif upper == 'ROWS':
            section = 'ROWS'
            continue
        elif upper == 'COLUMNS':
            section = 'COLUMNS'
            continue
        elif upper == 'RHS':
            section = 'RHS'
            continue
        elif upper == 'RANGES':
            section = 'RANGES'
            continue
        elif upper == 'BOUNDS':
            section = 'BOUNDS'
            continue
        elif upper == 'ENDATA':
            break

        # Process section data
        parts = line.split()

        if section == 'OBJSENSE':
            if 'MAX' in upper:
                obj_sense = -1

        elif section == 'ROWS':
            if len(parts) >= 2:
                rtype, rname = parts[0], parts[1]
                row_names.append(rname)
                row_types.append(rtype.upper())
                if rtype.upper() == 'N' and obj_name is None:
                    obj_name = rname

        elif section == 'COLUMNS':
            # COLUMNS section: colname rowname value [rowname value]
            if len(parts) >= 3:
                col_name = parts[0]
                if col_name not in col_name_to_idx:
                    col_name_to_idx[col_name] = len(col_names)
                    col_names.append(col_name)
                col_idx = col_name_to_idx[col_name]

                # Parse pairs
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        row_name = parts[i]
                        val = float(parts[i + 1])
                        if row_name in [rn for rn in row_names]:
                            row_idx = row_names.index(row_name)
                            row_indices.append(row_idx)
                            col_indices.append(col_idx)
                            values.append(val)

        elif section == 'RHS':
            if len(parts) >= 3:
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        row_name = parts[i]
                        val = float(parts[i + 1])
                        rhs[row_name] = val

        elif section == 'BOUNDS':
            if len(parts) >= 3:
                bound_type = parts[0].upper()
                bound_name = parts[1]
                col_name = parts[2]
                val = float(parts[3]) if len(parts) > 3 else 0.0

                if col_name not in bounds_lb:
                    bounds_lb[col_name] = 0.0
                    bounds_ub[col_name] = np.inf

                if bound_type == 'LO':
                    bounds_lb[col_name] = val
                elif bound_type == 'UP':
                    bounds_ub[col_name] = val
                elif bound_type == 'FX':
                    bounds_lb[col_name] = val
                    bounds_ub[col_name] = val
                elif bound_type == 'FR':
                    bounds_lb[col_name] = -np.inf
                    bounds_ub[col_name] = np.inf
                elif bound_type == 'MI':
                    bounds_lb[col_name] = -np.inf
                elif bound_type == 'PL':
                    bounds_ub[col_name] = np.inf
                elif bound_type == 'BV':
                    bounds_lb[col_name] = 0.0
                    bounds_ub[col_name] = 1.0

    # Build matrices
    n_vars = len(col_names)
    n_rows = len(row_names)

    if n_vars == 0:
        raise ValueError("No variables found in MPS file")

    # Create full coefficient matrix
    A_full = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_rows, n_vars)
    ).tocsr()

    # Separate objective from constraints
    obj_idx = None
    if obj_name:
        for i, (rn, rt) in enumerate(zip(row_names, row_types)):
            if rn == obj_name and rt == 'N':
                obj_idx = i
                break

    if obj_idx is not None:
        c = obj_sense * np.array(A_full[obj_idx].todense()).flatten()

        # Remove objective row
        keep_rows = [i for i in range(n_rows) if i != obj_idx]
        A = A_full[keep_rows]
        row_names_filtered = [row_names[i] for i in keep_rows]
        row_types_filtered = [row_types[i] for i in keep_rows]
    else:
        c = np.zeros(n_vars)
        A = A_full
        row_names_filtered = row_names
        row_types_filtered = row_types

    n_cons = A.shape[0]

    # Build RHS and sense vectors
    b = np.zeros(n_cons)
    sense = np.zeros(n_cons, dtype=int)

    for i, (rn, rt) in enumerate(zip(row_names_filtered, row_types_filtered)):
        b[i] = rhs.get(rn, 0.0)
        if rt == 'L':
            sense[i] = -1  # <=
        elif rt == 'G':
            sense[i] = 1   # >=
        elif rt == 'E':
            sense[i] = 0   # =

    # Build bounds
    lb = np.array([bounds_lb.get(cn, 0.0) for cn in col_names])
    ub = np.array([bounds_ub.get(cn, np.inf) for cn in col_names])

    return LPData(
        c=c,
        A=A,
        b=b,
        sense=sense,
        lb=lb,
        ub=ub,
        name=name,
        var_names=col_names,
        con_names=row_names_filtered,
    )


def load_problem(filepath: str) -> LPData:
    """
    Load LP problem from any supported format.

    Determines format from extension.

    Args:
        filepath: Path to problem file

    Returns:
        LPData structure
    """
    base = filepath.lower()

    # Strip compression extensions for format detection
    for ext in ['.gz', '.bz2']:
        if base.endswith(ext):
            base = base[:-len(ext)]

    if base.endswith('.mps'):
        return load_mps(filepath)
    elif base.endswith('.lp'):
        # LP format - not implemented yet, convert to MPS
        raise NotImplementedError("LP format not yet supported, use MPS")
    elif base.endswith('.mtx'):
        # Matrix Market - just the A matrix, no LP structure
        raise NotImplementedError("Matrix Market format needs additional files for LP")
    else:
        # Try MPS as default
        return load_mps(filepath)


def get_features(lp: LPData) -> Dict[str, Any]:
    """
    Extract features from an LP for use in conditional configuration.

    Args:
        lp: LP data structure

    Returns:
        Feature dictionary
    """
    A = lp.A if lp.A is not None else lp.A_eq

    if A is None or A.shape[0] == 0:
        return {
            "n_vars": lp.n_vars,
            "n_cons": 0,
            "nnz": 0,
            "density": 0.0,
            "coef_range_log10": 0.0,
        }

    # Basic sizes
    n_vars = A.shape[1]
    n_cons = A.shape[0]
    nnz = A.nnz
    density = nnz / (n_vars * n_cons) if n_vars > 0 and n_cons > 0 else 0.0

    # Coefficient statistics
    data = A.data
    if len(data) > 0:
        abs_data = np.abs(data)
        nonzero = abs_data[abs_data > 0]
        if len(nonzero) > 0:
            min_abs = nonzero.min()
            max_abs = abs_data.max()
            coef_range = np.log10(max_abs / min_abs) if min_abs > 0 else 0.0
        else:
            coef_range = 0.0
    else:
        coef_range = 0.0

    # Row/column densities
    row_nnz = np.diff(A.indptr)
    col_csr = A.tocsc()
    col_nnz = np.diff(col_csr.indptr)

    # Constraint type distribution
    if lp.sense is not None:
        n_leq = np.sum(lp.sense == -1)
        n_geq = np.sum(lp.sense == 1)
        n_eq = np.sum(lp.sense == 0)
    else:
        n_leq = n_geq = 0
        n_eq = n_cons

    return {
        "n_vars": int(n_vars),
        "n_cons": int(n_cons),
        "nnz": int(nnz),
        "density": float(density),
        "coef_range_log10": float(coef_range),
        "avg_row_nnz": float(row_nnz.mean()) if len(row_nnz) > 0 else 0.0,
        "max_row_nnz": int(row_nnz.max()) if len(row_nnz) > 0 else 0,
        "avg_col_nnz": float(col_nnz.mean()) if len(col_nnz) > 0 else 0.0,
        "max_col_nnz": int(col_nnz.max()) if len(col_nnz) > 0 else 0,
        "n_leq": int(n_leq),
        "n_geq": int(n_geq),
        "n_eq": int(n_eq),
    }
