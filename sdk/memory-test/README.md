# Memory-Test: Memvid Integration Verification

This directory contains a comprehensive test of the evolve-sdk memory system's integration with memvid-sdk. This is **not a unit test** - it's a realistic usage scenario that demonstrates the memory system working as intended.

## Why This Test Exists

The original memvid integration was a stub that fell back to JSON. This test was created to:
1. Verify memvid-sdk is actually being used (not JSON fallback)
2. Demonstrate semantic search works across process restarts
3. Test persistence - data survives when the process exits
4. Document real outputs as evidence

## Test Components

### 1. `scenario_evolution_run.py`
Simulates a realistic evolution run that:
- Records 5 DRIs (Don't Repeat Incidents) representing common ML mistakes
- Tests semantic search (finding "divergence" DRI when searching "loss became infinity")
- Records 5 submissions (2 successful, 3 failed)
- Records 2 model evaluations with proper sample sizes
- Tests sample size validation (rejects small sample sizes)
- Runs pre-action safety checks before submissions

### 2. `verify_persistence.py`
Runs in a **separate process** to verify data persists:
- Opens the same .mv2 store file
- Verifies all records are still there (5 DRIs, 5 submissions, 2 evaluations)
- Tests that semantic search still works
- Confirms data integrity

## Running the Tests

```bash
# From the sdk directory, use the virtual environment
cd sdk

# Run the evolution scenario
.venv/bin/python memory-test/scenario_evolution_run.py

# Verify persistence (run in separate process)
.venv/bin/python memory-test/verify_persistence.py
```

## Actual Test Output (January 17, 2026)

### Scenario Run Output

```
======================================================================
  MEMORY-TEST: Realistic Evolution Run Scenario
======================================================================

Memvid-SDK Available: True

Test project directory: .../sdk/memory-test/test_project

--- Initializing MemoryManager ---
Store path: .../test_project/.evolve-sdk/evolution_run/project_memory.mv2
Backend: memvid
Initial record counts: {'dris': 0, 'submissions': 0, 'evaluations': 0, 'config_changes': 0}

[OK] Using memvid backend

======================================================================
  PHASE 1: Recording Incidents (DRIs)
======================================================================
Added: DRI-001 - Learning rate too high caused divergence
Added: DRI-002 - Submitted wrong model checkpoint
Added: DRI-003 - Training data had duplicate entries
Added: DRI-004 - Evaluation used wrong metric normalization
Added: DRI-005 - Huggingface upload corrupted tokenizer config

Total DRIs recorded: 5

======================================================================
  PHASE 2: Testing Semantic Search
======================================================================

--- Search: 'model training issues gradient' ---
  Found: DRI-001 - Learning rate too high caused divergence
  Found: DRI-002 - Submitted wrong model checkpoint
  Found: DRI-003 - Training data had duplicate entries

--- Search: 'uploading model to cloud' ---
  Found: DRI-005 - Huggingface upload corrupted tokenizer config
  Found: DRI-001 - Learning rate too high caused divergence
  Found: DRI-002 - Submitted wrong model checkpoint

--- Search: 'preparing submission package' ---
  Found: DRI-002 - Submitted wrong model checkpoint
  Found: DRI-005 - Huggingface upload corrupted tokenizer config
  Found: DRI-003 - Training data had duplicate entries

--- Search: 'loss became infinity' (semantic for 'divergence') ---
  Found: DRI-001 - Learning rate too high caused divergence

======================================================================
  PHASE 3: Recording Submissions
======================================================================
[FAIL] sub-001: models/gen5_v1 - acc=0.45
[FAIL] sub-002: models/gen8_v1 - acc=0.62
[OK] sub-003: models/gen12_v2 - acc=0.78
[FAIL] sub-004: models/gen15_v1 - acc=0.71
[OK] sub-005: models/gen18_v3 - acc=0.82

Total submissions: 5
Successful: 2
Failed: 3

======================================================================
  PHASE 4: Recording Model Evaluations
======================================================================
Recorded: gen12_v2
  Metrics: {'accuracy': 0.78, 'f1': 0.75, 'precision': 0.8, 'recall': 0.71}
  Sample size: 500
  Verdict: Significant improvement over baseline (0.65 -> 0.78)
Recorded: gen18_v3
  Metrics: {'accuracy': 0.82, 'f1': 0.8, 'precision': 0.83, 'recall': 0.77}
  Sample size: 500
  Verdict: New best model. Ready for final submission.

--- Testing sample size validation (should fail) ---
[OK] Correctly rejected: Evaluation failed validation:
  - Sample size 10 < minimum 100. Conclusions from small samples are unreliable.

======================================================================
  PHASE 5: Pre-Action Safety Checks
======================================================================

--- Check before 'submission' action ---
Relevant DRIs: 4
Warnings: 4
  ! DRI-002: Submitted wrong model checkpoint (wasted 1 submissions)
  ! DRI-005: Huggingface upload corrupted tokenizer config (wasted 3 submissions)
  ! DRI-001: Learning rate too high caused divergence (wasted 2 submissions)
  ! DRI-003: Training data had duplicate entries (wasted 1 submissions)
Safe to proceed: True

--- Check before 'huggingface_upload' action ---
Relevant DRIs: 4
  ! DRI-005: Huggingface upload corrupted tokenizer config (wasted 3 submissions)
  ! DRI-001: Learning rate too high caused divergence (wasted 2 submissions)
  ! DRI-002: Submitted wrong model checkpoint (wasted 1 submissions)
  ! DRI-003: Training data had duplicate entries (wasted 1 submissions)

======================================================================
  PHASE 6: Final Statistics
======================================================================
Store path: .../test_project/.evolve-sdk/evolution_run/project_memory.mv2
Backend: memvid
Total records: 12
Counts by type:
  dris: 5
  submissions: 5
  evaluations: 2
  config_changes: 0

======================================================================
  TEST COMPLETE
======================================================================

Memvid backend: memvid
Store file: .../test_project/.evolve-sdk/evolution_run/project_memory.mv2
```

### Persistence Verification Output

```
======================================================================
  PERSISTENCE VERIFICATION
======================================================================

Memvid-SDK Available: True

[OK] Memvid store file exists: .../test_project/.evolve-sdk/evolution_run/project_memory.mv2
     Size: 116,439 bytes

======================================================================
  Loading Data in New Process
======================================================================
Backend: memvid
Total records: 12
Counts: {'dris': 5, 'submissions': 5, 'evaluations': 2, 'config_changes': 0}

======================================================================
  Verification Results
======================================================================
[PASS] dris: 5 (expected 5)
[PASS] submissions: 5 (expected 5)
[PASS] evaluations: 2 (expected 2)

======================================================================
  Data Integrity Check
======================================================================

DRIs loaded: 5
  - DRI-005: Huggingface upload corrupted tokenizer config...
  - DRI-004: Evaluation used wrong metric normalization...
  - DRI-003: Training data had duplicate entries...

Submissions loaded: 5
  Successful: 2
  Failed: 3

Evaluations loaded: 2
  - gen18_v3: accuracy=0.82
  - gen12_v2: accuracy=0.78

======================================================================
  Semantic Search Verification
======================================================================

Searching for 'gradient explosion'...
  Found 2 results:
    - DRI-001: Learning rate too high caused divergence
  [PASS] Found semantically related DRI

======================================================================
  PERSISTENCE TEST RESULT
======================================================================

[PASS] All persistence checks passed!
Data successfully survives process restarts.
```

## Key Verification Points

| Feature | Status | Evidence |
|---------|--------|----------|
| Memvid backend used | PASS | `Backend: memvid` in output |
| Store file is .mv2 | PASS | `project_memory.mv2` (116KB) |
| DRIs persist | PASS | 5 DRIs loaded in new process |
| Submissions persist | PASS | 5 submissions loaded in new process |
| Evaluations persist | PASS | 2 evaluations loaded in new process |
| Semantic search works | PASS | "gradient explosion" found "divergence" DRI |
| Sample size validation | PASS | Rejected sample_size=10 |
| Pre-action checks | PASS | Found relevant DRIs for submission/upload |

## Bugs Found and Fixed During Testing

### Bug: `load_all()` returned empty list after process restart

**Symptom**: Semantic search worked, but `load_all()` returned 0 records.

**Root Cause**: `timeline()` returns a truncated `preview` field, not the full text. The `preview` was too short to parse the complete JSON.

**Fix**: Changed `load_all()` to use `find()` for each schema type instead of `timeline()`. The `find()` API returns full text content.

**Location**: `sdk/evolve_sdk/memory/store.py:1444`

## Store File Structure

The test creates this directory structure:
```
memory-test/
├── test_project/
│   └── .evolve-sdk/
│       └── evolution_run/
│           └── project_memory.mv2   # 116KB memvid store
├── scenario_evolution_run.py
├── verify_persistence.py
├── debug_memvid.py      # API exploration scripts
├── debug_memvid2.py
├── debug_memvid3.py
├── debug_memvid4.py
└── README.md
```

## Conclusion

The memvid-sdk integration is **working correctly**:
1. Data is stored in `.mv2` format (not JSON fallback)
2. Data persists across process restarts
3. Semantic search finds related concepts (not just keyword matching)
4. Validation rules are enforced (sample size minimums)
5. Pre-action safety checks work as expected

The previous stub implementation has been replaced with real memvid integration.
