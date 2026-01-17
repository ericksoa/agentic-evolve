#!/usr/bin/env python3
"""Debug script to understand memvid's API and data structure."""

import sys
from pathlib import Path

sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

import memvid_sdk

store_path = Path(__file__).parent / "test_project" / ".evolve-sdk" / "evolution_run" / "project_memory.mv2"

print(f"Store exists: {store_path.exists()}")
print(f"Store size: {store_path.stat().st_size:,} bytes")

# Open store
print("\nOpening store...")
mem = memvid_sdk.use("basic", str(store_path), enable_lex=True)

# Check stats
print("\n--- stats() output ---")
stats = mem.stats()
print(f"stats = {stats}")
print(f"type = {type(stats)}")

# Check timeline
print("\n--- timeline() output ---")
timeline = mem.timeline(limit=20)
print(f"timeline type = {type(timeline)}")
print(f"timeline length = {len(timeline)}")
if timeline:
    print(f"First entry keys: {timeline[0].keys() if isinstance(timeline[0], dict) else 'not a dict'}")
    print(f"First entry: {timeline[0]}")

# Try find
print("\n--- find() output ---")
response = mem.find("learning rate", k=5)
print(f"find response type = {type(response)}")
print(f"find response keys = {response.keys() if hasattr(response, 'keys') else 'N/A'}")
if isinstance(response, dict) and "hits" in response:
    print(f"Number of hits: {len(response['hits'])}")
    if response['hits']:
        hit = response['hits'][0]
        print(f"First hit keys: {hit.keys()}")
        print(f"First hit text (first 200 chars): {hit.get('text', '')[:200]}")
