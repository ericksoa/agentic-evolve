#!/usr/bin/env python3
"""
Check training status on Lightning.ai studio.

Usage:
    python check_studio.py [STUDIO_NAME] [--lines N]

Examples:
    python check_studio.py                    # Check chess-llm-v4 (default)
    python check_studio.py chess-llm-v5
    python check_studio.py --lines 50         # Show more log lines
"""

import os
import sys
import re
import argparse

# Load env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from lightning_sdk import Studio

DEFAULT_STUDIO = "chess-llm-v4"


def check_studio(studio_name: str, log_lines: int = 30):
    """Check training status on the studio."""

    print("=" * 60)
    print(f"STUDIO: {studio_name}")
    print("=" * 60)

    # Connect to studio
    studio = Studio(
        name=studio_name,
        teamspace=os.environ.get("LIGHTNING_TEAMSPACE"),
        user=os.environ.get("LIGHTNING_AI_USERNAME"),
    )

    # 1. Status
    print(f"\n[1] Status: {studio.status}")
    print(f"    Machine: {studio.machine}")

    if "Running" not in str(studio.status):
        print("\n    Studio is not running!")
        return

    # 2. GPU Status
    print("\n[2] GPU Status:")
    try:
        gpu_out = studio.run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv")
        for line in gpu_out.strip().split('\n'):
            print(f"    {line}")
    except Exception as e:
        print(f"    Error: {e}")

    # 3. Running processes
    print("\n[3] Training Processes:")
    try:
        proc_out = studio.run("ps aux | grep python | grep -v grep | grep -v jupyter")
        if proc_out.strip():
            for line in proc_out.strip().split('\n'):
                parts = line.split()
                if len(parts) > 10:
                    pid = parts[1]
                    cpu = parts[2]
                    cmd = ' '.join(parts[10:])[:60]
                    print(f"    PID {pid} | CPU {cpu}% | {cmd}")
        else:
            print("    No training processes running!")
    except Exception as e:
        print(f"    Error: {e}")

    # 4. Screen sessions
    print("\n[4] Screen Sessions:")
    try:
        screen_out = studio.run("screen -ls 2>/dev/null || echo 'None'")
        for line in screen_out.strip().split('\n'):
            if line.strip() and 'Socket' not in line and 'There is' not in line:
                print(f"    {line.strip()}")
    except Exception as e:
        print(f"    Error: {e}")

    # 5. Find latest log and show progress
    print(f"\n[5] Training Progress:")
    try:
        # Find most recent log
        log_file = studio.run("ls -t /teamspace/studios/this_studio/*.log 2>/dev/null | head -1").strip()

        if log_file:
            print(f"    Log: {os.path.basename(log_file)}")

            # Get last N lines
            tail_out = studio.run(f"tail -{log_lines} {log_file}")

            # Parse progress - find ALL progress bars, pick the one with largest total (main training)
            all_progress = re.findall(r'(\d+)%\|[█▉▊▋▌▍▎▏ ]+\|\s*(\d+)/(\d+)', tail_out)
            progress_match = None
            if all_progress:
                # Pick the progress bar with the largest total (main training loop)
                progress_match = max(all_progress, key=lambda x: int(x[2]))

            loss_match = re.search(r"'loss':\s*([\d.]+)", tail_out)
            eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", tail_out)
            epoch_match = re.search(r"'epoch':\s*([\d.]+)", tail_out)
            acc_match = re.search(r"'eval_mean_token_accuracy':\s*([\d.]+)", tail_out)

            print("\n    --- Summary ---")
            if progress_match:
                pct, current, total = progress_match
                remaining = int(total) - int(current)
                eta_hours = remaining * 1.4 / 3600  # Estimate ~1.4s per step
                print(f"    Steps: {current}/{total} ({pct}%)")
                print(f"    Remaining: {remaining} steps (~{eta_hours:.1f}h)")
            if epoch_match:
                print(f"    Epoch: {epoch_match.group(1)}")
            if loss_match:
                print(f"    Train Loss: {loss_match.group(1)}")
            if eval_loss_match:
                print(f"    Eval Loss: {eval_loss_match.group(1)}")
            if acc_match:
                print(f"    Token Accuracy: {float(acc_match.group(1))*100:.1f}%")

            print("\n    --- Last 10 Lines ---")
            lines = [l for l in tail_out.split('\n') if l.strip()]
            for line in lines[-10:]:
                print(f"    {line[:90]}")
        else:
            print("    No log files found")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Check Lightning studio training status")
    parser.add_argument("studio", nargs="?", default=DEFAULT_STUDIO,
                        help=f"Studio name (default: {DEFAULT_STUDIO})")
    parser.add_argument("--lines", "-n", type=int, default=30,
                        help="Number of log lines to parse (default: 30)")

    args = parser.parse_args()
    check_studio(args.studio, args.lines)


if __name__ == "__main__":
    main()
