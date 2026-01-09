#!/usr/bin/env python3
"""
Run Best-of-N on Lightning.ai cloud compute.

This script launches multiple lightning.ai studios to run the Rust solver
in parallel, then aggregates results for Best-of-1000.

Usage:
    python run_bestof_lightning.py --runs 1000 --studios 10
    python run_bestof_lightning.py --runs 100 --studios 1  # Single studio test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

try:
    from lightning_sdk import Studio, Machine
except ImportError:
    print("Error: lightning-sdk not installed", file=sys.stderr)
    print("Run: pip install lightning-sdk", file=sys.stderr)
    sys.exit(1)


def run_bestof_on_studio(studio_id: int, runs_per_studio: int, max_n: int = 200) -> dict:
    """
    Run Best-of-N on a single lightning.ai studio.

    Returns dict with best packings for each n.
    """
    username = os.environ.get("LIGHTNING_AI_USERNAME")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")

    if not username or not teamspace:
        return {"error": "Lightning.ai credentials not set", "studio_id": studio_id}

    studio_name = f"santa-bestof-{studio_id}"
    print(f"[Studio {studio_id}] Starting {studio_name}...")

    try:
        # Create and start studio with CPU (no GPU needed for packing)
        studio = Studio(name=studio_name, teamspace=teamspace, user=username, create_ok=True)
        studio.start(Machine.CPU)
        print(f"[Studio {studio_id}] Studio started")

        # Install Rust and dependencies
        print(f"[Studio {studio_id}] Installing Rust...")
        studio.run_with_exit_code("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null")
        studio.run_with_exit_code("source ~/.cargo/env")

        # Upload Rust source code via tarball
        tarball_path = Path(__file__).parent / "rust_solver.tar.gz"

        if not tarball_path.exists():
            return {"error": "rust_solver.tar.gz not found - run: tar -czvf rust_solver.tar.gz rust/", "studio_id": studio_id}

        print(f"[Studio {studio_id}] Uploading Rust code tarball...")

        import base64

        # Read and encode tarball
        tarball_data = tarball_path.read_bytes()
        tarball_b64 = base64.b64encode(tarball_data).decode()

        # Upload via base64 (split into chunks if needed)
        chunk_size = 50000  # ~50KB chunks
        chunks = [tarball_b64[i:i+chunk_size] for i in range(0, len(tarball_b64), chunk_size)]

        studio.run_with_exit_code("rm -f ~/rust_solver.tar.gz.b64")
        for i, chunk in enumerate(chunks):
            studio.run_with_exit_code(f"echo -n '{chunk}' >> ~/rust_solver.tar.gz.b64")

        # Decode and extract
        studio.run_with_exit_code("base64 -d ~/rust_solver.tar.gz.b64 > ~/rust_solver.tar.gz")
        studio.run_with_exit_code("cd ~ && tar -xzf rust_solver.tar.gz")
        studio.run_with_exit_code("mv ~/rust ~/santa-packing")

        # Build
        print(f"[Studio {studio_id}] Building Rust project...")
        build_output, build_code = studio.run_with_exit_code(
            "cd ~/santa-packing && source ~/.cargo/env && cargo build --release --bin parallel_best_of_n 2>&1"
        )

        if build_code != 0:
            return {"error": f"Build failed: {build_output[:500]}", "studio_id": studio_id}

        print(f"[Studio {studio_id}] Running Best-of-{runs_per_studio}...")

        # Run best-of-N
        run_output, run_code = studio.run_with_exit_code(
            f"cd ~/santa-packing && source ~/.cargo/env && "
            f"./target/release/parallel_best_of_n {max_n} {runs_per_studio} 2>&1"
        )

        # Parse the score from output
        score = None
        for line in run_output.split('\n'):
            if 'PARALLEL_BEST_OF_N_SCORE=' in line:
                score = float(line.split('=')[1].strip(']'))
                break

        # Get the packings (we'd need to export them)
        # For now, just return the score
        result = {
            "studio_id": studio_id,
            "runs": runs_per_studio,
            "score": score,
            "output": run_output[-2000:] if len(run_output) > 2000 else run_output
        }

        print(f"[Studio {studio_id}] Complete - Score: {score}")
        return result

    except Exception as e:
        return {"error": str(e), "studio_id": studio_id}

    finally:
        # Stop studio to save credits
        try:
            studio.stop()
            print(f"[Studio {studio_id}] Studio stopped")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run Best-of-N on Lightning.ai")
    parser.add_argument('--runs', type=int, default=1000, help='Total runs (default: 1000)')
    parser.add_argument('--studios', type=int, default=10, help='Number of parallel studios (default: 10)')
    parser.add_argument('--max-n', type=int, default=200, help='Max n value (default: 200)')
    parser.add_argument('--output', type=str, default='bestof_lightning_results.json', help='Output file')

    args = parser.parse_args()

    runs_per_studio = args.runs // args.studios
    print(f"Running Best-of-{args.runs} using {args.studios} studios ({runs_per_studio} runs each)")
    print(f"Max n: {args.max_n}")
    print()

    start_time = time.time()
    results = []

    # Run studios in parallel
    with ThreadPoolExecutor(max_workers=args.studios) as executor:
        futures = {
            executor.submit(run_bestof_on_studio, i, runs_per_studio, args.max_n): i
            for i in range(args.studios)
        }

        for future in as_completed(futures):
            studio_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                if 'error' in result:
                    print(f"[Studio {studio_id}] ERROR: {result['error']}")
            except Exception as e:
                print(f"[Studio {studio_id}] Exception: {e}")
                results.append({"error": str(e), "studio_id": studio_id})

    elapsed = time.time() - start_time

    # Aggregate results
    valid_results = [r for r in results if 'score' in r and r['score'] is not None]

    if valid_results:
        best_score = min(r['score'] for r in valid_results)
        avg_score = sum(r['score'] for r in valid_results) / len(valid_results)

        print()
        print("=" * 60)
        print(f"RESULTS: Best-of-{args.runs} on Lightning.ai")
        print("=" * 60)
        print(f"Studios used: {args.studios}")
        print(f"Valid results: {len(valid_results)}/{args.studios}")
        print(f"Best score: {best_score:.4f}")
        print(f"Avg score: {avg_score:.4f}")
        print(f"Time: {elapsed:.1f}s")
        print("=" * 60)
    else:
        print("No valid results!")

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            "total_runs": args.runs,
            "studios": args.studios,
            "runs_per_studio": runs_per_studio,
            "elapsed_seconds": elapsed,
            "results": results,
            "best_score": best_score if valid_results else None,
            "avg_score": avg_score if valid_results else None,
        }, f, indent=2)

    print(f"Results saved to {args.output}")

    return 0 if valid_results else 1


if __name__ == "__main__":
    sys.exit(main())
