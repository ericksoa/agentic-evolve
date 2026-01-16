"""
policy.py - Genome policy representation for cuOpt LP AutoTuner.

A policy is a human-readable JSON structure that maps:
  LP features -> (transforms, solver params, batching decisions)

The policy uses bucketed decision trees: simple threshold-based rules
that can be easily evolved through mutation and crossover.

Policy structure:
{
    "version": "1.0",
    "name": "policy_name",

    "transforms": {
        "enable_row_scale": true/false or rule,
        "enable_col_scale": true/false or rule,
        "enable_row_reorder": true/false or rule,
        "enable_col_reorder": true/false or rule,
        "enable_rhs_normalize": true/false or rule,
        "enable_bound_tighten": true/false or rule,
        "row_reorder_method": "degree_ascending" or rule,
        "col_reorder_method": "degree_descending" or rule,
    },

    "solver_params": {
        "tolerance": float or rule,
        "max_iterations": int or rule,
        "presolve": bool or rule,
        "scaling_method": str or rule,
        ...
    },

    "batching": {
        "enabled": true/false,
        "size_thresholds": [100, 1000, 10000],  # bucket boundaries
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 32,
        "strategy": "size_homogeneous" | "density_homogeneous" | "mixed",
    }
}

A "rule" is a conditional expression:
{
    "type": "conditional",
    "feature": "size_bucket",  # feature name
    "conditions": [
        {"match": "tiny", "value": true},
        {"match": "small", "value": true},
        {"default": false}
    ]
}
"""

import json
import random
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


# Default baseline policy (cuOpt defaults, no transforms)
BASELINE_POLICY = {
    "version": "1.0",
    "name": "baseline",
    "description": "Baseline policy: no transforms, default solver params, no batching",

    "transforms": {
        "enable_row_scale": False,
        "enable_col_scale": False,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-6,
        "max_iterations": 10000,
        "presolve": True,
        "scaling_method": "none",
        "threads": 1,
    },

    "batching": {
        "enabled": False,
        "size_thresholds": [100, 1000, 10000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 1,
        "strategy": "none",
    },
}


# All mutable genes and their domains
GENE_DOMAINS = {
    # Transform toggles
    "transforms.enable_row_scale": {"type": "bool"},
    "transforms.enable_col_scale": {"type": "bool"},
    "transforms.enable_row_reorder": {"type": "bool"},
    "transforms.enable_col_reorder": {"type": "bool"},
    "transforms.enable_rhs_normalize": {"type": "bool"},
    "transforms.enable_bound_tighten": {"type": "bool"},

    # Transform methods
    "transforms.row_reorder_method": {
        "type": "categorical",
        "values": ["degree_ascending", "degree_descending", "rhs_ascending"],
    },
    "transforms.col_reorder_method": {
        "type": "categorical",
        "values": ["degree_ascending", "degree_descending", "bounds_first"],
    },

    # Solver params
    "solver_params.tolerance": {
        "type": "log_float",
        "min": 1e-9,
        "max": 1e-3,
    },
    "solver_params.max_iterations": {
        "type": "int",
        "min": 1000,
        "max": 100000,
    },
    "solver_params.presolve": {"type": "bool"},
    "solver_params.scaling_method": {
        "type": "categorical",
        "values": ["none", "geometric", "equilibration"],
    },
    "solver_params.threads": {
        "type": "int",
        "min": 1,
        "max": 8,
    },

    # Batching
    "batching.enabled": {"type": "bool"},
    "batching.size_thresholds": {
        "type": "threshold_list",
        "min_val": 10,
        "max_val": 100000,
        "length": 3,
    },
    "batching.density_thresholds": {
        "type": "threshold_list",
        "min_val": 0.001,
        "max_val": 0.5,
        "length": 2,
    },
    "batching.max_batch_size": {
        "type": "int",
        "min": 1,
        "max": 64,
    },
    "batching.strategy": {
        "type": "categorical",
        "values": ["none", "size_homogeneous", "density_homogeneous", "mixed"],
    },
}


def get_nested(d: Dict, path: str, default=None):
    """Get a nested dictionary value by dot-separated path."""
    keys = path.split('.')
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(d: Dict, path: str, value):
    """Set a nested dictionary value by dot-separated path."""
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


@dataclass
class Policy:
    """
    Policy genome for LP AutoTuner.

    Wraps a JSON policy dict and provides methods for:
    - Loading/saving
    - Evaluating decisions based on LP features
    - Mutation and crossover for evolution
    """
    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(BASELINE_POLICY))

    @classmethod
    def baseline(cls) -> 'Policy':
        """Create the baseline (default) policy."""
        return cls(copy.deepcopy(BASELINE_POLICY))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'Policy':
        """Load policy from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create policy from dictionary."""
        return cls(copy.deepcopy(data))

    def to_json(self, path: Union[str, Path]):
        """Save policy to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Return policy as dictionary."""
        return copy.deepcopy(self.data)

    @property
    def name(self) -> str:
        return self.data.get('name', 'unnamed')

    @name.setter
    def name(self, value: str):
        self.data['name'] = value

    def get_value(self, path: str, features: Optional[Dict[str, Any]] = None):
        """
        Get a policy value, potentially evaluating a conditional rule.

        Args:
            path: Dot-separated path to the value
            features: LP features for conditional evaluation

        Returns:
            The resolved value
        """
        raw_value = get_nested(self.data, path)

        # If it's a conditional rule, evaluate it
        if isinstance(raw_value, dict) and raw_value.get('type') == 'conditional':
            return self._evaluate_conditional(raw_value, features)

        return raw_value

    def _evaluate_conditional(self, rule: Dict, features: Optional[Dict]) -> Any:
        """Evaluate a conditional rule against features."""
        if features is None:
            # Return default if no features provided
            for cond in rule.get('conditions', []):
                if 'default' in cond:
                    return cond['default']
            return None

        feature_name = rule.get('feature')
        feature_value = features.get(feature_name)

        for cond in rule.get('conditions', []):
            if 'match' in cond:
                if feature_value == cond['match']:
                    return cond['value']
            elif 'lt' in cond:
                if feature_value < cond['lt']:
                    return cond['value']
            elif 'gt' in cond:
                if feature_value > cond['gt']:
                    return cond['value']
            elif 'default' in cond:
                return cond['default']

        return None

    def get_transform_config(self, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get transform configuration resolved for given features."""
        return {
            'enable_row_scale': self.get_value('transforms.enable_row_scale', features),
            'enable_col_scale': self.get_value('transforms.enable_col_scale', features),
            'enable_row_reorder': self.get_value('transforms.enable_row_reorder', features),
            'enable_col_reorder': self.get_value('transforms.enable_col_reorder', features),
            'enable_rhs_normalize': self.get_value('transforms.enable_rhs_normalize', features),
            'enable_bound_tighten': self.get_value('transforms.enable_bound_tighten', features),
            'row_reorder_method': self.get_value('transforms.row_reorder_method', features),
            'col_reorder_method': self.get_value('transforms.col_reorder_method', features),
        }

    def get_solver_params(self, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get solver parameters resolved for given features."""
        return {
            'tolerance': self.get_value('solver_params.tolerance', features),
            'max_iterations': self.get_value('solver_params.max_iterations', features),
            'presolve': self.get_value('solver_params.presolve', features),
            'scaling_method': self.get_value('solver_params.scaling_method', features),
            'threads': self.get_value('solver_params.threads', features),
        }

    def get_batching_config(self, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get batching configuration resolved for given features."""
        return {
            'enabled': self.get_value('batching.enabled', features),
            'size_thresholds': self.get_value('batching.size_thresholds', features),
            'density_thresholds': self.get_value('batching.density_thresholds', features),
            'max_batch_size': self.get_value('batching.max_batch_size', features),
            'strategy': self.get_value('batching.strategy', features),
        }


def mutate_policy(policy: Policy, mutation_rate: float = 0.2, rng: Optional[random.Random] = None) -> Policy:
    """
    Create a mutated copy of a policy.

    Args:
        policy: Parent policy
        mutation_rate: Probability of mutating each gene
        rng: Random number generator (for reproducibility)

    Returns:
        Mutated policy copy
    """
    if rng is None:
        rng = random.Random()

    new_policy = Policy.from_dict(policy.to_dict())
    new_policy.name = f"{policy.name}_mut"

    for gene_path, domain in GENE_DOMAINS.items():
        if rng.random() > mutation_rate:
            continue

        current_value = get_nested(new_policy.data, gene_path)

        # Don't mutate conditional rules (for now, keep them as-is)
        if isinstance(current_value, dict) and current_value.get('type') == 'conditional':
            continue

        new_value = _mutate_gene(current_value, domain, rng)
        set_nested(new_policy.data, gene_path, new_value)

    return new_policy


def _mutate_gene(current_value, domain: Dict, rng: random.Random):
    """Mutate a single gene according to its domain."""
    gene_type = domain['type']

    if gene_type == 'bool':
        return not current_value

    elif gene_type == 'categorical':
        values = domain['values']
        return rng.choice(values)

    elif gene_type == 'int':
        # Gaussian mutation clamped to range
        min_val, max_val = domain['min'], domain['max']
        sigma = (max_val - min_val) * 0.1
        new_val = current_value + rng.gauss(0, sigma)
        return int(max(min_val, min(max_val, round(new_val))))

    elif gene_type == 'float':
        min_val, max_val = domain['min'], domain['max']
        sigma = (max_val - min_val) * 0.1
        new_val = current_value + rng.gauss(0, sigma)
        return max(min_val, min(max_val, new_val))

    elif gene_type == 'log_float':
        # Mutate in log space
        import math
        min_val, max_val = domain['min'], domain['max']
        log_current = math.log10(current_value) if current_value > 0 else math.log10(min_val)
        log_min, log_max = math.log10(min_val), math.log10(max_val)
        sigma = (log_max - log_min) * 0.1
        log_new = log_current + rng.gauss(0, sigma)
        log_new = max(log_min, min(log_max, log_new))
        return 10 ** log_new

    elif gene_type == 'threshold_list':
        # Mutate individual thresholds
        min_val, max_val = domain['min_val'], domain['max_val']
        length = domain['length']
        new_list = []
        for v in current_value:
            if rng.random() < 0.5:
                # Mutate this threshold
                sigma = (max_val - min_val) * 0.1
                new_v = v + rng.gauss(0, sigma)
                new_v = max(min_val, min(max_val, new_v))
            else:
                new_v = v
            new_list.append(new_v)
        # Keep sorted
        new_list.sort()
        return new_list

    return current_value


def crossover_policies(parent1: Policy, parent2: Policy, rng: Optional[random.Random] = None) -> Policy:
    """
    Create a child policy by crossing over two parents.

    Uses uniform crossover at the gene level.

    Args:
        parent1, parent2: Parent policies
        rng: Random number generator

    Returns:
        Child policy
    """
    if rng is None:
        rng = random.Random()

    child = Policy.from_dict(parent1.to_dict())
    child.name = f"{parent1.name}x{parent2.name}"

    for gene_path in GENE_DOMAINS.keys():
        if rng.random() < 0.5:
            # Take from parent2
            value = get_nested(parent2.data, gene_path)
            set_nested(child.data, gene_path, copy.deepcopy(value))

    return child


def random_policy(rng: Optional[random.Random] = None) -> Policy:
    """
    Generate a completely random policy.

    Args:
        rng: Random number generator

    Returns:
        Random policy
    """
    if rng is None:
        rng = random.Random()

    policy = Policy.baseline()
    policy.name = f"random_{rng.randint(0, 99999):05d}"

    for gene_path, domain in GENE_DOMAINS.items():
        value = _random_gene(domain, rng)
        set_nested(policy.data, gene_path, value)

    return policy


def _random_gene(domain: Dict, rng: random.Random):
    """Generate a random value for a gene."""
    gene_type = domain['type']

    if gene_type == 'bool':
        return rng.choice([True, False])

    elif gene_type == 'categorical':
        return rng.choice(domain['values'])

    elif gene_type == 'int':
        return rng.randint(domain['min'], domain['max'])

    elif gene_type == 'float':
        return rng.uniform(domain['min'], domain['max'])

    elif gene_type == 'log_float':
        import math
        log_min = math.log10(domain['min'])
        log_max = math.log10(domain['max'])
        return 10 ** rng.uniform(log_min, log_max)

    elif gene_type == 'threshold_list':
        values = [rng.uniform(domain['min_val'], domain['max_val'])
                  for _ in range(domain['length'])]
        values.sort()
        return values

    return None


def load_policy(path_or_name: str) -> Policy:
    """
    Load a policy from file or return baseline.

    Args:
        path_or_name: 'baseline', 'random', or path to JSON file

    Returns:
        Policy instance
    """
    if path_or_name == 'baseline':
        return Policy.baseline()
    elif path_or_name == 'random':
        return random_policy()
    else:
        return Policy.from_json(path_or_name)
