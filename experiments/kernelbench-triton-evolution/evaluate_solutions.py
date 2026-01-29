#!/usr/bin/env python3
"""
Evaluate the fitness of mutation solutions.
"""

import ast
import sys
import time
import json
from pathlib import Path
import importlib.util
import traceback

def analyze_solution_structure(file_path):
    """Analyze the code structure without executing."""
    try:
        content = Path(file_path).read_text()
        tree = ast.parse(content)

        functions = []
        imports = []
        has_triton_decorator = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        has_triton_decorator = '@triton.jit' in content or 'triton.jit' in content
        has_triton_import = any('triton' in imp for imp in imports)
        has_softmax_func = any(name in ['softmax_triton', 'triton_softmax', 'softmax', 'kernel']
                              for name in functions)

        return {
            'valid_structure': has_triton_import and has_softmax_func and has_triton_decorator,
            'functions': functions,
            'imports': imports,
            'has_triton': has_triton_import,
            'has_softmax_func': has_softmax_func,
            'has_kernel': has_triton_decorator,
            'content_length': len(content),
        }

    except Exception as e:
        return {
            'valid_structure': False,
            'error': str(e),
            'functions': [],
            'imports': []
        }

def load_and_test_solution(file_path):
    """Load and test a solution for correctness."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("solution", file_path)
        module = importlib.util.module_from_spec(spec)

        # Redirect stdout to capture any prints
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            spec.loader.exec_module(module)

        # Find the softmax function
        softmax_fn = None
        for name in ['softmax_triton', 'triton_softmax', 'softmax', 'kernel']:
            if hasattr(module, name):
                softmax_fn = getattr(module, name)
                break

        if softmax_fn is None:
            return {
                'valid': False,
                'error': 'No softmax function found',
                'fitness': 0
            }

        # Test correctness with small input (CPU or CUDA)
        import torch

        try:
            # Try CUDA first
            if torch.cuda.is_available():
                device = 'cuda'
                x = torch.randn(2, 8, device=device, dtype=torch.float32)
            else:
                device = 'cpu'
                x = torch.randn(2, 8, device=device, dtype=torch.float32)

            expected = torch.softmax(x, dim=-1)
            result = softmax_fn(x)

            # Check correctness
            if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
                return {
                    'valid': False,
                    'error': 'Incorrect output vs PyTorch softmax',
                    'fitness': 0,
                    'max_diff': (result - expected).abs().max().item()
                }

            # Check properties
            row_sums = result.sum(dim=-1)
            sum_error = (row_sums - 1.0).abs().max().item()

            if sum_error > 1e-4:
                return {
                    'valid': False,
                    'error': f'Row sums not 1: max error {sum_error}',
                    'fitness': 0
                }

            if torch.isnan(result).any() or torch.isinf(result).any():
                return {
                    'valid': False,
                    'error': 'Contains NaN or Inf',
                    'fitness': 0
                }

            if (result < 0).any():
                return {
                    'valid': False,
                    'error': 'Contains negative values',
                    'fitness': 0
                }

            return {
                'valid': True,
                'error': None,
                'fitness': 1.0,  # Base fitness for correct implementation
                'max_diff': (result - expected).abs().max().item(),
                'sum_error': sum_error,
                'device_used': device
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Runtime error: {str(e)}',
                'fitness': 0,
                'device_used': device if 'device' in locals() else 'unknown'
            }

    except Exception as e:
        return {
            'valid': False,
            'error': f'Failed to load module: {str(e)}',
            'fitness': 0
        }

def evaluate_solution(file_path):
    """Complete evaluation of a solution."""
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            'file': str(file_path),
            'valid': False,
            'fitness': 0,
            'error': 'File not found',
            'notes': 'File does not exist'
        }

    # Structure analysis
    structure = analyze_solution_structure(file_path)

    if not structure.get('valid_structure', False):
        return {
            'file': str(file_path),
            'valid': False,
            'fitness': 0,
            'error': f'Invalid structure: {structure.get("error", "Missing required components")}',
            'notes': f'Found functions: {structure.get("functions", [])}',
            'structure': structure
        }

    # Functional testing
    test_result = load_and_test_solution(file_path)

    return {
        'file': str(file_path),
        'valid': test_result['valid'],
        'fitness': test_result['fitness'],
        'error': test_result.get('error'),
        'notes': f"Max diff: {test_result.get('max_diff', 'N/A')}, Device: {test_result.get('device_used', 'unknown')}",
        'max_diff': test_result.get('max_diff'),
        'sum_error': test_result.get('sum_error'),
        'device_used': test_result.get('device_used'),
        'structure': structure
    }

if __name__ == "__main__":
    solutions = [
        ".evolve-sdk/evolve_fastest_triton_softmax/mutations/gen2a.py",
        ".evolve-sdk/evolve_fastest_triton_softmax/mutations/gen2b.py",
        ".evolve-sdk/evolve_fastest_triton_softmax/mutations/gen2x.py"
    ]

    evaluations = []
    valid_solutions = []

    for solution in solutions:
        result = evaluate_solution(solution)
        evaluations.append(result)

        if result['valid']:
            valid_solutions.append(result)

    # Determine best solution
    best = None
    if valid_solutions:
        # For correctness-only eval, all valid solutions have fitness 1.0
        # Choose the one with smallest numerical error
        best = min(valid_solutions, key=lambda x: x.get('max_diff', float('inf')))

    # Ranking by fitness (valid first, then by error)
    ranking = sorted(evaluations, key=lambda x: (-x['fitness'], x.get('max_diff') or float('inf')))

    result = {
        "evaluations": evaluations,
        "best": {"file": best['file'], "fitness": best['fitness']} if best else None,
        "ranking": [r['file'] for r in ranking],
        "summary": {
            "total_solutions": len(evaluations),
            "valid_solutions": len(valid_solutions),
            "cuda_available": "Yes" if __import__('torch').cuda.is_available() else "No"
        }
    }

    print(json.dumps(result, indent=2))