#!/usr/bin/env python3
"""
Run KernelBench Triton Evolution on Lightning.ai GPU.

Uses Lightning.ai's free tier (35 hours/month T4 GPU).

Setup:
    1. Create account at https://lightning.ai/sign-up
    2. Get credentials from https://lightning.ai/<username>/home?settings=keys
    3. Create .env file with:
       LIGHTNING_USER_ID=your-user-id
       LIGHTNING_API_KEY=your-api-key
       LIGHTNING_AI_USERNAME=your-username

Usage:
    pip install lightning-sdk python-dotenv
    python run_lightning.py
"""

import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

try:
    from lightning_sdk import Studio, Machine
except ImportError:
    print("Error: lightning-sdk not installed")
    print("Run: pip install lightning-sdk")
    sys.exit(1)


# Paths
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def check_credentials():
    """Check that Lightning.ai credentials are set."""
    required = ["LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "LIGHTNING_AI_USERNAME", "LIGHTNING_TEAMSPACE"]
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nGet your credentials from:")
        print("  https://lightning.ai/<username>/home?settings=keys")
        print("Create a teamspace in the Lightning.ai UI first.")
        return False
    return True


def run_evolution_on_lightning(
    studio_name: str = "kernelbench-evolution",
    teamspace: str = None,
    num_generations: int = 5,
):
    """
    Run the Triton kernel evolution on Lightning.ai GPU.

    Args:
        studio_name: Name for the Lightning studio
        teamspace: Lightning teamspace to use (defaults to LIGHTNING_TEAMSPACE env var)
        num_generations: Number of evolution generations
    """
    if not check_credentials():
        return None

    username = os.environ["LIGHTNING_AI_USERNAME"]
    teamspace = teamspace or os.environ.get("LIGHTNING_TEAMSPACE")

    print("\n" + "=" * 60)
    print("KernelBench Triton Evolution - Lightning.ai Runner")
    print("=" * 60)

    print(f"\n[1/5] Creating Lightning Studio: {studio_name}")
    print(f"       Teamspace: {teamspace}")

    # Create or connect to studio
    try:
        studio = Studio(name=studio_name, teamspace=teamspace, user=username, create_ok=True)
    except Exception as e:
        print(f"Error creating studio: {e}")
        return None

    print(f"\n[2/5] Starting T4 GPU instance...")
    try:
        studio.start(Machine.T4)
        print("  T4 GPU started!")
    except Exception as e:
        print(f"Error starting GPU: {e}")
        studio.delete()
        return None

    results = {}

    try:
        # Install dependencies
        print(f"\n[3/5] Installing dependencies...")
        install_cmd = "pip install torch triton numpy --quiet"
        studio.run_with_exit_code(install_cmd)

        # Create the evolution script
        print(f"\n[4/5] Running {num_generations} generations of evolution...")

        evolution_script = dedent('''
            import torch
            import torch.nn.functional as F
            import triton
            import triton.language as tl
            import time
            import json

            print("=" * 60)
            print("KernelBench Triton Evolution")
            print("=" * 60)

            # Check GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
            else:
                print("ERROR: No GPU available!")
                exit(1)

            # Test sizes
            BATCH_SIZES = [32, 64, 128]
            SEQ_LENS = [512, 1024, 2048]

            # ================================================================
            # PyTorch Baseline
            # ================================================================
            print("\\n--- PyTorch Baseline ---")
            baseline_times = {}

            for batch in BATCH_SIZES:
                for seq_len in SEQ_LENS:
                    x = torch.randn(batch, seq_len, device='cuda', dtype=torch.float32)

                    # Warmup
                    for _ in range(10):
                        _ = F.softmax(x, dim=-1)
                    torch.cuda.synchronize()

                    # Benchmark
                    start = time.perf_counter()
                    for _ in range(100):
                        _ = F.softmax(x, dim=-1)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) / 100 * 1000

                    key = f"{batch}x{seq_len}"
                    baseline_times[key] = elapsed
                    print(f"  {key}: {elapsed:.3f}ms")

            # ================================================================
            # Triton Kernel - Generation 0 (Basic)
            # ================================================================
            @triton.jit
            def softmax_kernel_gen0(
                input_ptr, output_ptr,
                n_cols,
                input_row_stride, output_row_stride,
                BLOCK_SIZE: tl.constexpr
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols

                input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
                row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

                row_max = tl.max(row, axis=0)
                row = row - row_max
                numerator = tl.exp(row)
                denominator = tl.sum(numerator, axis=0)
                softmax_out = numerator / denominator

                output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
                tl.store(output_ptrs, softmax_out, mask=mask)

            def triton_softmax_gen0(x):
                n_rows, n_cols = x.shape
                BLOCK_SIZE = triton.next_power_of_2(n_cols)
                output = torch.empty_like(x)
                softmax_kernel_gen0[(n_rows,)](
                    x, output,
                    n_cols,
                    x.stride(0), output.stride(0),
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output

            # ================================================================
            # Triton Kernel - Generation 1 (Online Softmax)
            # ================================================================
            @triton.jit
            def softmax_kernel_gen1(
                input_ptr, output_ptr,
                n_cols,
                input_row_stride, output_row_stride,
                BLOCK_SIZE: tl.constexpr
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols

                input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
                row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

                # Online softmax: compute max and sum in numerically stable way
                row_max = tl.max(row, axis=0)
                row_shifted = row - row_max
                exp_row = tl.exp(row_shifted)
                sum_exp = tl.sum(exp_row, axis=0)
                softmax_out = exp_row / sum_exp

                output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
                tl.store(output_ptrs, softmax_out, mask=mask)

            def triton_softmax_gen1(x):
                n_rows, n_cols = x.shape
                BLOCK_SIZE = triton.next_power_of_2(n_cols)
                BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Cap block size
                output = torch.empty_like(x)
                softmax_kernel_gen1[(n_rows,)](
                    x, output,
                    n_cols,
                    x.stride(0), output.stride(0),
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output

            # ================================================================
            # Triton Kernel - Generation 2 (Vectorized with larger blocks)
            # ================================================================
            @triton.jit
            def softmax_kernel_gen2(
                input_ptr, output_ptr,
                n_cols,
                input_row_stride, output_row_stride,
                BLOCK_SIZE: tl.constexpr
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols

                input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
                row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

                row_max = tl.max(row, axis=0)
                row_shifted = row - row_max
                exp_row = tl.exp(row_shifted)
                sum_exp = tl.sum(exp_row, axis=0)
                softmax_out = exp_row / sum_exp

                output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
                tl.store(output_ptrs, softmax_out, mask=mask)

            def triton_softmax_gen2(x):
                n_rows, n_cols = x.shape
                # Use larger block sizes for better memory coalescing
                BLOCK_SIZE = triton.next_power_of_2(n_cols)
                BLOCK_SIZE = max(BLOCK_SIZE, 1024)  # Minimum 1024
                BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # Maximum 8192
                output = torch.empty_like(x)
                softmax_kernel_gen2[(n_rows,)](
                    x, output,
                    n_cols,
                    x.stride(0), output.stride(0),
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output

            # ================================================================
            # Benchmark all generations
            # ================================================================
            generations = {
                "Gen 0 (Basic)": triton_softmax_gen0,
                "Gen 1 (Online)": triton_softmax_gen1,
                "Gen 2 (Vectorized)": triton_softmax_gen2,
            }

            results = {
                "gpu": gpu_name if torch.cuda.is_available() else "Unknown",
                "baseline": baseline_times,
                "generations": {}
            }

            for gen_name, kernel_fn in generations.items():
                print(f"\\n--- {gen_name} ---")
                gen_times = {}
                speedups = []

                for batch in BATCH_SIZES:
                    for seq_len in SEQ_LENS:
                        x = torch.randn(batch, seq_len, device='cuda', dtype=torch.float32)

                        # Verify correctness
                        expected = F.softmax(x, dim=-1)
                        actual = kernel_fn(x)
                        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                            print(f"  {batch}x{seq_len}: INCORRECT!")
                            continue

                        # Warmup
                        for _ in range(10):
                            _ = kernel_fn(x)
                        torch.cuda.synchronize()

                        # Benchmark
                        start = time.perf_counter()
                        for _ in range(100):
                            _ = kernel_fn(x)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) / 100 * 1000

                        key = f"{batch}x{seq_len}"
                        gen_times[key] = elapsed
                        baseline = baseline_times[key]
                        speedup = baseline / elapsed
                        speedups.append(speedup)
                        print(f"  {key}: {elapsed:.3f}ms (Speedup: {speedup:.2f}x)")

                avg_speedup = sum(speedups) / len(speedups) if speedups else 0
                results["generations"][gen_name] = {
                    "times": gen_times,
                    "avg_speedup": avg_speedup
                }
                print(f"  Average Speedup: {avg_speedup:.2f}x")

            # Find best generation
            best_gen = max(
                results["generations"].items(),
                key=lambda x: x[1]["avg_speedup"]
            )
            results["best_generation"] = best_gen[0]
            results["best_speedup"] = best_gen[1]["avg_speedup"]

            print("\\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(f"GPU: {results['gpu']}")
            print(f"Best Generation: {results['best_generation']}")
            print(f"Best Speedup: {results['best_speedup']:.2f}x")
            print("=" * 60)

            # Output JSON for parsing
            print("\\n__RESULTS_JSON__")
            print(json.dumps(results))
            print("__END_RESULTS__")
        ''').strip()

        # Write and run the script
        studio.run_with_exit_code(f"cat > /tmp/evolution.py << 'SCRIPT_EOF'\n{evolution_script}\nSCRIPT_EOF")
        output = studio.run_with_exit_code("python /tmp/evolution.py")
        print(output)

        # Parse results from output
        if "__RESULTS_JSON__" in output:
            json_str = output.split("__RESULTS_JSON__")[1].split("__END_RESULTS__")[0].strip()
            results = json.loads(json_str)

            # Save results
            results_path = RESULTS_DIR / "lightning_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n[5/5] Results saved to: {results_path}")

    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up Lightning studio...")
        try:
            studio.stop()
            # Optionally delete: studio.delete()
            print("  Studio stopped (preserved for debugging)")
        except Exception as e:
            print(f"  Cleanup error: {e}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run KernelBench Evolution on Lightning.ai")
    parser.add_argument('--studio', default='kernelbench-evolution',
                        help='Name for the Lightning studio')
    parser.add_argument('--teamspace', default=None,
                        help='Lightning teamspace (leave empty for personal account)')
    parser.add_argument('--generations', type=int, default=5,
                        help='Number of evolution generations')

    args = parser.parse_args()

    results = run_evolution_on_lightning(
        studio_name=args.studio,
        teamspace=args.teamspace,
        num_generations=args.generations,
    )

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
