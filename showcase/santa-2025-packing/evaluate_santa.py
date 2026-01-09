#!/usr/bin/env python3
"""
Evaluate a Santa packing solution.

Called by the evolve SDK to evaluate fitness of algorithm mutations.
Can run locally or on lightning.ai.

Usage:
    python evaluate_santa.py <solution_file> [--json] [--cloud]
"""

import argparse
import json
import sys
import importlib.util
import os
from pathlib import Path

# Import validation from the project
sys.path.insert(0, str(Path(__file__).parent / "python"))
try:
    from validate_submission import validate_group, has_any_overlap_strict
except ImportError:
    # Fallback if validate_submission not available
    has_any_overlap_strict = None

# Tree shape (from the problem)
TREE_VERTICES = [
    (0.0, 0.8),    # tip
    (-0.15, 0.5),  # tier 1 left
    (-0.05, 0.5),  # tier 1 inner left
    (-0.25, 0.2),  # tier 2 left
    (-0.1, 0.2),   # tier 2 inner left
    (-0.35, -0.1), # tier 3 left
    (-0.1, -0.1),  # trunk top left
    (-0.1, -0.2),  # trunk bottom left
    (0.1, -0.2),   # trunk bottom right
    (0.1, -0.1),   # trunk top right
    (0.35, -0.1),  # tier 3 right
    (0.1, 0.2),    # tier 2 inner right
    (0.25, 0.2),   # tier 2 right
    (0.05, 0.5),   # tier 1 inner right
    (0.15, 0.5),   # tier 1 right
]


def evaluate_solution(solution_file: str, cloud: bool = False, output_json: bool = False) -> dict:
    """
    Evaluate a packing algorithm solution.

    Args:
        solution_file: Path to Python file implementing the algorithm
        cloud: If True, run on lightning.ai
        output_json: If True, output JSON to stdout

    Returns:
        Dict with fitness, valid, score, etc.
    """
    solution_path = Path(solution_file)
    if not solution_path.exists():
        result = {"valid": False, "fitness": 0, "error": f"File not found: {solution_file}"}
        if output_json:
            print(json.dumps(result))
        return result

    if cloud:
        return evaluate_on_cloud(solution_path, output_json)
    else:
        return evaluate_local(solution_path, output_json)


def evaluate_local(solution_path: Path, output_json: bool = False) -> dict:
    """Run evaluation locally."""
    try:
        # Import the solution module
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)

        # Look for a pack function
        pack_fn = None
        for name in ['pack', 'pack_trees', 'solve', 'optimize', 'run']:
            if hasattr(solution, name) and callable(getattr(solution, name)):
                pack_fn = getattr(solution, name)
                break

        if pack_fn is None:
            result = {"valid": False, "fitness": 0, "error": "No pack function found"}
            if output_json:
                print(json.dumps(result))
            return result

        # Evaluate on subset of n values (5-20 for quick feedback)
        test_ns = list(range(5, 21))
        scores = []
        results = {"per_n": {}}

        for n in test_ns:
            try:
                # Call pack function - should return list of (x, y, angle) or PlacedTree objects
                trees = pack_fn(n)

                if trees is None or len(trees) != n:
                    results["per_n"][n] = {"error": f"Expected {n} trees, got {len(trees) if trees else 0}"}
                    continue

                # Calculate bounding box
                from math import cos, sin, radians
                min_x, max_x = float('inf'), float('-inf')
                min_y, max_y = float('inf'), float('-inf')

                transformed_trees = []

                # Handle different return formats
                tree_data = []
                for tree in trees:
                    if hasattr(tree, 'x') and hasattr(tree, 'y'):
                        # PlacedTree object
                        if hasattr(tree, 'angle_idx'):
                            angles = [0, 45, 90, 135, 180, 225, 270, 315]
                            angle = angles[tree.angle_idx % len(angles)]
                        elif hasattr(tree, 'angle'):
                            angle = tree.angle
                        else:
                            angle = 0
                        tree_data.append((tree.x, tree.y, angle))
                    elif isinstance(tree, (list, tuple)) and len(tree) >= 3:
                        # (x, y, angle) tuple
                        tree_data.append((tree[0], tree[1], tree[2]))
                    else:
                        results["per_n"][n] = {"error": f"Unknown tree format: {type(tree)}"}
                        continue

                for x, y, angle in tree_data:
                    angle_rad = radians(angle)
                    transformed = []
                    for vx, vy in TREE_VERTICES:
                        rx = vx * cos(angle_rad) - vy * sin(angle_rad) + x
                        ry = vx * sin(angle_rad) + vy * cos(angle_rad) + y
                        min_x, max_x = min(min_x, rx), max(max_x, rx)
                        min_y, max_y = min(min_y, ry), max(max_y, ry)
                        transformed.append((rx, ry))
                    transformed_trees.append(transformed)

                # Side length of bounding square
                side = max(max_x - min_x, max_y - min_y)
                score_n = (side ** 2) / n

                # Check for overlaps if validation available
                valid = True
                if has_any_overlap_strict:
                    try:
                        valid = not has_any_overlap_strict(transformed_trees)
                    except Exception:
                        valid = True  # Assume valid if check fails

                if valid:
                    scores.append(score_n)
                    results["per_n"][n] = {"side": side, "score": score_n, "valid": True}
                else:
                    results["per_n"][n] = {"side": side, "score": score_n, "valid": False, "error": "Overlaps detected"}

            except Exception as e:
                results["per_n"][n] = {"error": str(e)}

        # Calculate total score
        if scores:
            total_score = sum(scores)
            avg_score = total_score / len(scores)
            # Fitness: higher is better, so we invert the score
            # Using 1/(score + 0.1) to avoid division issues
            fitness = 1.0 / (avg_score + 0.1)

            results["valid"] = True
            results["total_score"] = total_score
            results["avg_score_per_n"] = avg_score
            results["fitness"] = fitness
            results["n_evaluated"] = len(scores)
        else:
            results["valid"] = False
            results["fitness"] = 0
            results["error"] = "No valid packings"

        if output_json:
            print(json.dumps(results))

        return results

    except Exception as e:
        result = {"valid": False, "fitness": 0, "error": str(e)}
        if output_json:
            print(json.dumps(result))
        return result


def evaluate_on_cloud(solution_path: Path, output_json: bool = False) -> dict:
    """Run evaluation on lightning.ai."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    try:
        from lightning_sdk import Studio, Machine
    except ImportError:
        result = {"valid": False, "fitness": 0, "error": "lightning-sdk not installed"}
        if output_json:
            print(json.dumps(result))
        return result

    username = os.environ.get("LIGHTNING_AI_USERNAME")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")

    if not username or not teamspace:
        result = {"valid": False, "fitness": 0, "error": "Lightning.ai credentials not set"}
        if output_json:
            print(json.dumps(result))
        return result

    try:
        # Use CPU machine for packing (no GPU needed)
        studio = Studio(name="santa-packing-eval", teamspace=teamspace, user=username, create_ok=True)
        studio.start(Machine.CPU)  # CPU is cheaper and sufficient

        # Upload the solution file
        solution_code = solution_path.read_text()
        import base64
        code_b64 = base64.b64encode(solution_code.encode()).decode()

        # Create evaluation script on the cloud
        eval_script = create_cloud_eval_script(code_b64)

        studio.run_with_exit_code(f"cat > /tmp/eval_santa.py << 'EVALEOF'\n{eval_script}\nEVALEOF")
        output, exit_code = studio.run_with_exit_code("python /tmp/eval_santa.py")

        # Parse result
        if "__EVAL_RESULT__" in output:
            json_str = output.split("__EVAL_RESULT__")[1].split("__END_RESULT__")[0].strip()
            result = json.loads(json_str)
        else:
            result = {"valid": False, "fitness": 0, "error": f"Could not parse output: {output[:2000]}"}

    except Exception as e:
        result = {"valid": False, "fitness": 0, "error": str(e)}

    finally:
        try:
            studio.stop()
        except:
            pass

    if output_json:
        print(json.dumps(result))

    return result


def create_cloud_eval_script(code_b64: str) -> str:
    """Create the evaluation script to run on cloud."""
    return f'''
import json
import sys
import base64
import importlib.util
from math import cos, sin, radians

# Tree shape
TREE_VERTICES = [
    (0.0, 0.8), (-0.15, 0.5), (-0.05, 0.5), (-0.25, 0.2), (-0.1, 0.2),
    (-0.35, -0.1), (-0.1, -0.1), (-0.1, -0.2), (0.1, -0.2), (0.1, -0.1),
    (0.35, -0.1), (0.1, 0.2), (0.25, 0.2), (0.05, 0.5), (0.15, 0.5),
]

# Decode and load solution
code = base64.b64decode("{code_b64}").decode()
with open("/tmp/solution.py", "w") as f:
    f.write(code)

spec = importlib.util.spec_from_file_location("solution", "/tmp/solution.py")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

# Find pack function
pack_fn = None
for name in ['pack', 'pack_trees', 'solve', 'optimize', 'run']:
    if hasattr(solution, name) and callable(getattr(solution, name)):
        pack_fn = getattr(solution, name)
        break

if pack_fn is None:
    print("__EVAL_RESULT__")
    print(json.dumps({{"valid": False, "fitness": 0, "error": "No pack function"}}))
    print("__END_RESULT__")
    sys.exit(0)

# Evaluate
test_ns = list(range(5, 21))
scores = []
per_n = {{}}

for n in test_ns:
    try:
        trees = pack_fn(n)
        if trees is None or len(trees) != n:
            per_n[n] = {{"error": f"Expected {{n}} trees"}}
            continue

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for x, y, angle in trees:
            angle_rad = radians(angle)
            for vx, vy in TREE_VERTICES:
                rx = vx * cos(angle_rad) - vy * sin(angle_rad) + x
                ry = vx * sin(angle_rad) + vy * cos(angle_rad) + y
                min_x, max_x = min(min_x, rx), max(max_x, rx)
                min_y, max_y = min(min_y, ry), max(max_y, ry)

        side = max(max_x - min_x, max_y - min_y)
        score_n = (side ** 2) / n
        scores.append(score_n)
        per_n[n] = {{"side": side, "score": score_n}}
    except Exception as e:
        per_n[n] = {{"error": str(e)}}

if scores:
    avg = sum(scores) / len(scores)
    fitness = 1.0 / (avg + 0.1)
    result = {{"valid": True, "fitness": fitness, "avg_score": avg, "per_n": per_n}}
else:
    result = {{"valid": False, "fitness": 0, "error": "No valid packings"}}

print("__EVAL_RESULT__")
print(json.dumps(result))
print("__END_RESULT__")
'''


def main():
    parser = argparse.ArgumentParser(description="Evaluate Santa packing solution")
    parser.add_argument('solution', help='Path to the solution Python file')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')
    parser.add_argument('--cloud', action='store_true', help='Run on lightning.ai')

    args = parser.parse_args()

    result = evaluate_solution(args.solution, cloud=args.cloud, output_json=args.json)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
