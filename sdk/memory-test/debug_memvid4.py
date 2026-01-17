#!/usr/bin/env python3
"""Test frame() with URI string."""

import sys
from pathlib import Path

sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

import memvid_sdk

store_path = Path(__file__).parent / "test_project" / ".evolve-sdk" / "evolution_run" / "project_memory.mv2"
mem = memvid_sdk.use("basic", str(store_path), enable_lex=True)

# Try frame() with URI string
print("--- Testing frame() with URI ---")
timeline = mem.timeline(limit=2)
for entry in timeline:
    uri = entry.get('uri')  # 'mv2://frames/0'
    print(f"\nURI: {uri}")
    try:
        full_frame = mem.frame(uri)
        print(f"  type: {type(full_frame)}")
        if isinstance(full_frame, dict):
            print(f"  keys: {list(full_frame.keys())}")
            text = full_frame.get('text', '')
            print(f"  text (first 300): {text[:300]}")
    except Exception as e:
        print(f"  error: {e}")
