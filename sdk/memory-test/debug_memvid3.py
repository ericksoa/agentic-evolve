#!/usr/bin/env python3
"""Debug script to test frame() method."""

import sys
from pathlib import Path

sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

import memvid_sdk

store_path = Path(__file__).parent / "test_project" / ".evolve-sdk" / "evolution_run" / "project_memory.mv2"
mem = memvid_sdk.use("basic", str(store_path), enable_lex=True)

# Try frame() method
print("--- Testing frame() method ---")
timeline = mem.timeline(limit=3)
for entry in timeline:
    frame_id = entry.get('frame_id')
    print(f"\nFrame {frame_id}:")
    print(f"  preview: {entry.get('preview', '')[:80]}...")

    try:
        full_frame = mem.frame(frame_id)
        print(f"  frame() type: {type(full_frame)}")
        if isinstance(full_frame, dict):
            print(f"  frame() keys: {full_frame.keys()}")
            text = full_frame.get('text', full_frame.get('content', ''))
            print(f"  text (first 200): {text[:200] if text else 'None'}...")
        else:
            print(f"  frame() result: {str(full_frame)[:300]}")
    except Exception as e:
        print(f"  frame() error: {e}")

# Alternative: search for each schema type and dedupe
print("\n\n--- Alternative: Load all via search ---")

def load_all_via_search(mem):
    """Load all frames by searching for each schema type."""
    schema_types = ["DRI", "Submission", "ModelEvaluation", "ConfigChange"]
    all_frames = {}  # Use dict to dedupe by _id

    for schema_type in schema_types:
        response = mem.find(schema_type, k=1000)
        hits = response.get('hits', [])
        for hit in hits:
            text = hit.get('text', '')
            if text.startswith('{'):
                import json
                try:
                    # Extract JSON
                    depth = 0
                    for i, char in enumerate(text):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                frame = json.loads(text[:i+1])
                                frame_id = frame.get('_id', str(hit.get('frame_id')))
                                all_frames[frame_id] = frame
                                break
                except json.JSONDecodeError:
                    pass

    return list(all_frames.values())

frames = load_all_via_search(mem)
print(f"Loaded {len(frames)} unique frames via search")
for frame in sorted(frames, key=lambda f: f.get('_id', '')):
    schema_type = frame.get('_schema_type', 'unknown')
    frame_id = frame.get('_id', 'N/A')
    if schema_type == 'DRI':
        print(f"  {frame_id}: {schema_type} - {frame.get('title', 'N/A')[:50]}")
    elif schema_type == 'Submission':
        print(f"  {frame_id}: {schema_type} - {frame.get('model_path', 'N/A')}")
    elif schema_type == 'ModelEvaluation':
        print(f"  {frame_id}: {schema_type} - {frame.get('model_name', 'N/A')}")
