#!/usr/bin/env python3
"""Debug script to find how to get full frame content."""

import sys
from pathlib import Path

sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

import memvid_sdk

store_path = Path(__file__).parent / "test_project" / ".evolve-sdk" / "evolution_run" / "project_memory.mv2"
mem = memvid_sdk.use("basic", str(store_path), enable_lex=True)

# Check available methods
print("Available methods on mem object:")
methods = [m for m in dir(mem) if not m.startswith('_')]
print(methods)

# Try to get a frame by ID
print("\n--- Trying to get frame by uri ---")
timeline = mem.timeline(limit=1)
if timeline:
    uri = timeline[0].get('uri')
    frame_id = timeline[0].get('frame_id')
    print(f"uri: {uri}")
    print(f"frame_id: {frame_id}")

    # Try various methods
    if hasattr(mem, 'get'):
        print("Has get() method")
        try:
            result = mem.get(uri)
            print(f"get(uri) result: {result}")
        except Exception as e:
            print(f"get(uri) failed: {e}")

    if hasattr(mem, 'read'):
        print("Has read() method")
        try:
            result = mem.read(frame_id)
            print(f"read(frame_id) result: {result}")
        except Exception as e:
            print(f"read(frame_id) failed: {e}")

# Try wildcard search to get all
print("\n--- Trying wildcard search ---")
try:
    # Use * as query to match all
    response = mem.find("*", k=100)
    print(f"find('*') hits: {len(response.get('hits', []))}")
except Exception as e:
    print(f"find('*') failed: {e}")

# Try empty query
print("\n--- Trying empty search ---")
try:
    response = mem.find("", k=100)
    print(f"find('') hits: {len(response.get('hits', []))}")
except Exception as e:
    print(f"find('') failed: {e}")

# Try searching for common words that should be in all frames
print("\n--- Trying to search for schema types ---")
for schema_type in ["DRI", "Submission", "ModelEvaluation"]:
    response = mem.find(schema_type, k=50)
    hits = response.get('hits', [])
    print(f"find('{schema_type}'): {len(hits)} hits")
