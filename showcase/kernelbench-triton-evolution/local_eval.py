#!/usr/bin/env python3
"""
Local evaluation for checking basic code structure without GPU.
"""
import ast
import sys
from pathlib import Path

def check_solution_structure(file_path):
    """Check if solution has proper structure without executing it."""
    try:
        content = Path(file_path).read_text()

        # Parse the AST to check structure
        tree = ast.parse(content)

        # Check for required functions
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        # Check if it's a proper Triton implementation
        has_triton = any('triton' in imp for imp in imports if imp)
        has_softmax_func = any(name in ['softmax_triton', 'triton_softmax', 'softmax', 'kernel']
                              for name in functions)
        has_kernel = any('@triton.jit' in content or 'triton.jit' in content)

        return {
            'valid': has_triton and has_softmax_func and has_kernel,
            'functions': functions,
            'imports': imports,
            'has_triton': has_triton,
            'has_softmax_func': has_softmax_func,
            'has_kernel': has_kernel,
            'content_length': len(content)
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'functions': [],
            'imports': []
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python local_eval.py <solution_file>")
        sys.exit(1)

    result = check_solution_structure(sys.argv[1])
    print(f"Analysis: {result}")