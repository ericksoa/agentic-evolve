#!/usr/bin/env python3
"""
Evaluate a Triton kernel on Lightning.ai GPU.

This script is called by the evolve SDK to evaluate fitness of mutations.
It uploads the kernel to Lightning.ai, runs benchmarks, and returns results.

Usage:
    python evaluate_on_lightning.py <solution_file> [--json]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

try:
    from lightning_sdk import Studio, Machine
except ImportError:
    print("Error: lightning-sdk not installed", file=sys.stderr)
    print("Run: pip install lightning-sdk", file=sys.stderr)
    sys.exit(1)


def evaluate_kernel(solution_file: str, output_json: bool = False) -> dict:
    """
    Evaluate a Triton kernel on Lightning.ai GPU.

    Args:
        solution_file: Path to the kernel Python file
        output_json: If True, output JSON to stdout

    Returns:
        Dict with fitness, valid, speedup, etc.
    """
    # Read the kernel code
    solution_path = Path(solution_file)
    if not solution_path.exists():
        result = {"valid": False, "fitness": 0, "error": f"File not found: {solution_file}"}
        if output_json:
            print(json.dumps(result))
        return result

    kernel_code = solution_path.read_text()

    # Get Lightning credentials
    username = os.environ.get("LIGHTNING_AI_USERNAME")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")

    if not username or not teamspace:
        result = {"valid": False, "fitness": 0, "error": "Lightning.ai credentials not set"}
        if output_json:
            print(json.dumps(result))
        return result

    # Connect to Lightning studio
    try:
        studio = Studio(name="kernelbench-eval", teamspace=teamspace, user=username, create_ok=True)
        studio.start(Machine.T4)
    except Exception as e:
        result = {"valid": False, "fitness": 0, "error": f"Could not start GPU: {e}"}
        if output_json:
            print(json.dumps(result))
        return result

    try:
        # Install dependencies (if not already)
        studio.run_with_exit_code("pip install torch triton --quiet 2>/dev/null || true")

        # Create evaluation script
        # First, write the kernel code to a file so Triton JIT can access source
        import base64
        kernel_code_b64 = base64.b64encode(kernel_code.encode()).decode()

        eval_script = dedent(f'''
            import torch
            import torch.nn.functional as F
            import time
            import json
            import sys
            import base64
            import importlib.util

            # The kernel code to evaluate (base64 encoded to avoid indentation issues)
            kernel_code = base64.b64decode("{kernel_code_b64}").decode()

            # Write kernel to a file so Triton JIT can access source via inspect
            with open("/tmp/triton_kernel.py", "w") as f:
                f.write(kernel_code)

            # Import the kernel module
            spec = importlib.util.spec_from_file_location("triton_kernel", "/tmp/triton_kernel.py")
            kernel_module = importlib.util.module_from_spec(spec)
            sys.modules["triton_kernel"] = kernel_module
            spec.loader.exec_module(kernel_module)

            # Find the softmax function in the imported module
            softmax_fn = None
            for name in ['softmax_triton', 'triton_softmax', 'softmax', 'kernel']:
                if hasattr(kernel_module, name) and callable(getattr(kernel_module, name)):
                    softmax_fn = getattr(kernel_module, name)
                    break

            if softmax_fn is None:
                print(json.dumps({{"valid": False, "fitness": 0, "error": "No softmax function found"}}))
                sys.exit(0)

            # Test sizes - use larger sizes where Triton can shine
            TEST_SIZES = [
                (256, 4096),
                (512, 4096),
                (1024, 2048),
                (1024, 4096),
            ]

            results = {{
                "valid": True,
                "fitness": 0,
                "speedups": {{}},
                "errors": [],
            }}

            speedups = []

            for batch, seq_len in TEST_SIZES:
                try:
                    x = torch.randn(batch, seq_len, device='cuda', dtype=torch.float32)

                    # Correctness check
                    expected = F.softmax(x, dim=-1)
                    try:
                        actual = softmax_fn(x)
                    except Exception as e:
                        results["errors"].append(f"{{batch}}x{{seq_len}}: {{str(e)}}")
                        continue

                    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                        results["errors"].append(f"{{batch}}x{{seq_len}}: Incorrect output")
                        continue

                    # Benchmark PyTorch
                    for _ in range(10):
                        _ = F.softmax(x, dim=-1)
                    torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(100):
                        _ = F.softmax(x, dim=-1)
                    torch.cuda.synchronize()
                    pytorch_time = (time.perf_counter() - start) / 100

                    # Benchmark Triton
                    for _ in range(10):
                        _ = softmax_fn(x)
                    torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(100):
                        _ = softmax_fn(x)
                    torch.cuda.synchronize()
                    triton_time = (time.perf_counter() - start) / 100

                    speedup = pytorch_time / triton_time
                    key = f"{{batch}}x{{seq_len}}"
                    results["speedups"][key] = speedup
                    speedups.append(speedup)

                except Exception as e:
                    results["errors"].append(f"{{batch}}x{{seq_len}}: {{str(e)}}")

            # Calculate fitness
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                # Fitness: 0 if slower than PyTorch, otherwise speedup
                results["fitness"] = avg_speedup if avg_speedup > 1.0 else avg_speedup * 0.1
                results["avg_speedup"] = avg_speedup
            else:
                results["valid"] = False
                results["fitness"] = 0

            print("__EVAL_RESULT__")
            print(json.dumps(results))
            print("__END_RESULT__")
        ''').strip()

        # Write and run eval script
        studio.run_with_exit_code(f"cat > /tmp/eval_kernel.py << 'EVALEOF'\n{eval_script}\nEVALEOF")
        output, exit_code = studio.run_with_exit_code("python /tmp/eval_kernel.py")

        # Parse result
        if "__EVAL_RESULT__" in output:
            json_str = output.split("__EVAL_RESULT__")[1].split("__END_RESULT__")[0].strip()
            result = json.loads(json_str)
        else:
            result = {"valid": False, "fitness": 0, "error": f"Could not parse output: {output[:2000]}"}

    except Exception as e:
        result = {"valid": False, "fitness": 0, "error": str(e)}

    finally:
        # Stop studio to save credits
        try:
            studio.stop()
        except:
            pass

    if output_json:
        print(json.dumps(result))

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Triton kernel on Lightning.ai GPU")
    parser.add_argument('solution', help='Path to the kernel Python file')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')
    parser.add_argument('--task', default='softmax', help='Task name (for compatibility)')

    args = parser.parse_args()

    result = evaluate_kernel(args.solution, output_json=args.json)

    # Exit code: 0 if valid, 1 if invalid
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
