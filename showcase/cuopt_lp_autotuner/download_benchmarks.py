#!/usr/bin/env python3
"""
download_benchmarks.py - Download standard LP benchmark problems.

Downloads Netlib LP benchmarks - the standard test suite used by LP solver
developers worldwide. These are real optimization problems from industry.

Usage:
    python3 -m showcase.cuopt_lp_autotuner.download_benchmarks --output ./benchmarks
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import List, Dict
import argparse

# Netlib LP benchmark problems
# Source: https://www.netlib.org/lp/data/
# These are classic LP benchmarks used to evaluate solver performance

NETLIB_BENCHMARKS = [
    # Small problems (< 100 vars) - good for quick tests
    {"name": "afiro", "vars": 32, "cons": 27, "description": "Small test problem"},
    {"name": "sc50a", "vars": 48, "cons": 50, "description": "Set covering"},
    {"name": "sc50b", "vars": 48, "cons": 50, "description": "Set covering"},
    {"name": "kb2", "vars": 41, "cons": 43, "description": "Basis test"},
    {"name": "blend", "vars": 83, "cons": 74, "description": "Blending problem"},

    # Medium problems (100-500 vars)
    {"name": "sc105", "vars": 103, "cons": 105, "description": "Set covering"},
    {"name": "sc205", "vars": 203, "cons": 205, "description": "Set covering"},
    {"name": "share1b", "vars": 225, "cons": 117, "description": "Share model"},
    {"name": "share2b", "vars": 79, "cons": 96, "description": "Share model"},
    {"name": "lotfi", "vars": 308, "cons": 153, "description": "Logistics"},
    {"name": "brandy", "vars": 249, "cons": 220, "description": "Refinery model"},
    {"name": "scorpion", "vars": 358, "cons": 388, "description": "Wildlife management"},
    {"name": "bandm", "vars": 472, "cons": 305, "description": "Band matrix"},
    {"name": "e226", "vars": 282, "cons": 223, "description": "Economic model"},
    {"name": "scfxm1", "vars": 457, "cons": 330, "description": "Network"},

    # Larger problems (500-2000 vars)
    {"name": "israel", "vars": 142, "cons": 174, "description": "Economic planning"},
    {"name": "agg", "vars": 163, "cons": 488, "description": "Aggregation"},
    {"name": "agg2", "vars": 302, "cons": 516, "description": "Aggregation 2"},
    {"name": "agg3", "vars": 302, "cons": 516, "description": "Aggregation 3"},
    {"name": "scsd1", "vars": 760, "cons": 77, "description": "Scheduling"},
    {"name": "sctap1", "vars": 480, "cons": 300, "description": "Traffic assignment"},
    {"name": "ship04s", "vars": 1458, "cons": 402, "description": "Shipping"},
    {"name": "ship08s", "vars": 2387, "cons": 778, "description": "Shipping large"},
    {"name": "boeing1", "vars": 384, "cons": 351, "description": "Aircraft production"},
    {"name": "boeing2", "vars": 143, "cons": 166, "description": "Aircraft production"},

    # Classic hard problems
    {"name": "stocfor1", "vars": 111, "cons": 117, "description": "Stochastic forestry"},
    {"name": "adlittle", "vars": 97, "cons": 56, "description": "Economic model"},
    {"name": "stair", "vars": 467, "cons": 356, "description": "Staircase problem"},
    {"name": "grow7", "vars": 301, "cons": 140, "description": "Growth model"},
    {"name": "grow15", "vars": 645, "cons": 300, "description": "Growth model large"},
    {"name": "grow22", "vars": 946, "cons": 440, "description": "Growth model xlarge"},
]

# Minimal benchmark set for quick testing
MINIMAL_BENCHMARKS = ["afiro", "blend", "sc105", "lotfi", "brandy", "israel", "agg"]

NETLIB_BASE_URL = "https://www.netlib.org/lp/data"

# Alternative source with clean MPS format (Hans Mittelmann's collection)
MITTELMANN_BASE_URL = "https://plato.asu.edu/ftp/lp/mps"

# Problems available in clean MPS format - from scipy test data
# These are decompressed Netlib problems in standard MPS format
CLEAN_MPS_SOURCES = {
    # Scipy test files (clean MPS format)
    "afiro": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/afiro.mps",
    "blend": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/blend.mps",
    "kb2": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/kb2.mps",
    "sc50a": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/sc50a.mps",
    "sc50b": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/sc50b.mps",
    "sc105": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/sc105.mps",
    "sc205": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/sc205.mps",
    "share1b": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/share1b.mps",
    "share2b": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/share2b.mps",
    "lotfi": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/lotfi.mps",
    "adlittle": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/adlittle.mps",
    "stocfor1": "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/tests/test_linprog_data/mps/stocfor1.mps",
}


def download_netlib_problem(name: str, output_dir: Path, verbose: bool = True) -> bool:
    """
    Download a single Netlib LP problem.

    Args:
        name: Problem name (e.g., "afiro")
        output_dir: Directory to save to
        verbose: Print progress

    Returns:
        True if successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.mps"
    temp_path = output_dir / f"{name}.tmp"

    # Try clean MPS source first (coin-or/Data)
    if name in CLEAN_MPS_SOURCES:
        try:
            if verbose:
                print(f"  Downloading {name}...", end=" ", flush=True)

            urllib.request.urlretrieve(CLEAN_MPS_SOURCES[name], output_path)

            if verbose:
                print("OK")
            return True
        except Exception as e:
            if output_path.exists():
                output_path.unlink()

    # Fall back to netlib (compressed format)
    urls_to_try = [
        f"{NETLIB_BASE_URL}/{name}.gz",
        f"{NETLIB_BASE_URL}/{name}",
    ]

    for url in urls_to_try:
        try:
            if verbose and name not in CLEAN_MPS_SOURCES:
                print(f"  Downloading {name}...", end=" ", flush=True)

            urllib.request.urlretrieve(url, temp_path)

            # Decompress if needed
            if url.endswith('.gz'):
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                temp_path.unlink()
            else:
                temp_path.rename(output_path)

            if verbose:
                print("OK")
            return True

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            continue

    if verbose:
        print("FAILED")
    return False


def download_benchmarks(
    output_dir: str = "./benchmarks",
    problems: List[str] = None,
    minimal: bool = False,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Download LP benchmark problems.

    Args:
        output_dir: Directory to save benchmarks
        problems: Specific problems to download (None = all)
        minimal: Only download minimal set for quick testing
        verbose: Print progress

    Returns:
        Dict mapping problem name to success status
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which problems to download
    if problems:
        to_download = problems
    elif minimal:
        to_download = MINIMAL_BENCHMARKS
    else:
        to_download = [b["name"] for b in NETLIB_BENCHMARKS]

    if verbose:
        print(f"Downloading {len(to_download)} Netlib LP benchmarks to {output_path}/")
        print()

    results = {}
    success_count = 0

    for name in to_download:
        success = download_netlib_problem(name, output_path, verbose)
        results[name] = success
        if success:
            success_count += 1

    if verbose:
        print()
        print(f"Downloaded {success_count}/{len(to_download)} benchmarks")
        print(f"Location: {output_path.absolute()}")

    return results


def list_benchmarks():
    """Print available benchmarks."""
    print("Available Netlib LP Benchmarks:")
    print()
    print(f"{'Name':<12} {'Vars':>6} {'Cons':>6}  Description")
    print("-" * 60)
    for b in NETLIB_BENCHMARKS:
        print(f"{b['name']:<12} {b['vars']:>6} {b['cons']:>6}  {b['description']}")
    print()
    print(f"Total: {len(NETLIB_BENCHMARKS)} problems")
    print(f"Minimal set ({len(MINIMAL_BENCHMARKS)}): {', '.join(MINIMAL_BENCHMARKS)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Netlib LP benchmark problems"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./benchmarks",
        help="Output directory (default: ./benchmarks)"
    )
    parser.add_argument(
        "--minimal", "-m",
        action="store_true",
        help="Only download minimal benchmark set"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks and exit"
    )
    parser.add_argument(
        "--problems", "-p",
        nargs="+",
        help="Specific problems to download"
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    download_benchmarks(
        output_dir=args.output,
        problems=args.problems,
        minimal=args.minimal,
    )


if __name__ == "__main__":
    main()
