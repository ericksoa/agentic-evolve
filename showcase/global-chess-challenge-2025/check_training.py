#!/usr/bin/env python3
"""Check training progress on Lightning.ai"""
import os
from dotenv import load_dotenv

load_dotenv()

# Lightning.ai credentials
os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

from lightning_sdk import Studio

LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

def check_progress():
    """Check training progress"""
    studio = Studio(
        name="chess-llm-v4",
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
    )

    print("=" * 60)
    print("Training Progress Check")
    print("=" * 60)
    print()

    # Check if process is running
    ps_output = studio.run("ps aux | grep train_chess_v2 | grep -v grep || echo 'NOT_RUNNING'")
    if "NOT_RUNNING" in ps_output:
        print("STATUS: Training process NOT running")
    else:
        print("STATUS: Training process is running")
    print()

    # Get last 100 lines of log
    print("Recent log output:")
    print("-" * 60)
    log_output = studio.run("tail -100 ~/train_chess_v2.log 2>/dev/null || echo 'No log yet'")
    print(log_output)

    # Check for saved checkpoints
    print()
    print("Checkpoints:")
    print("-" * 60)
    checkpoints = studio.run("ls -la ~/chess-training/models-v2/ 2>/dev/null || echo 'No checkpoints yet'")
    print(checkpoints)

if __name__ == "__main__":
    check_progress()
