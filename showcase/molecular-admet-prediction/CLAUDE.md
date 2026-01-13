# Molecular ADMET Prediction - Project Instructions

## CRITICAL: Trust Validation is MANDATORY

**NEVER accept evolution results without running the full trust validation pipeline.**

When evolving solutions in this project:

1. **Use the SDK** - Run evolution through `python -m evolve_sdk`, NOT by manually creating mutations and running `evaluate.py` directly
2. **If you must run manually** - You MUST also run adversary validation:
   ```bash
   ./venv/bin/python escalated_validation.py <solution.py>
   ```
3. **All champions require trust dossier** - Check `.evolve-sdk/*/trust_dossier.md` exists

### Why This Matters
This is a drug discovery project. False positives in hERG toxicity prediction could lead to unsafe drugs. Results must be:
- Reproducible (variance gates)
- Not gaming the evaluator (adversary review)
- Documented (trust dossier)
- Human-verified for borderline cases

## How to Resume Work

When asked to "resume work" or "continue evolution":

1. **Check current state**:
   ```bash
   cat .evolve-sdk/evolve_herg_molecular_property/evolution.json
   ```

2. **Identify the champion** and its trust status

3. **If trust validation is missing**, run it BEFORE continuing:
   ```bash
   ./venv/bin/python escalated_validation.py <champion.py>
   ```

4. **Continue evolution using SDK**, not manual mutation

## Project Structure

- `evolve_config.json` - Evolution configuration (includes trust settings)
- `.evolve-sdk/` - Evolution state and mutations
- `evaluate.py` - Basic fitness evaluation (ROC-AUC)
- `escalated_validation.py` - Full trust validation (L1-L3 tests)
- `adversary_validation.py` - Adversary agent checks

## Current State (Update After Each Session)

**Champion**: `gen12c.py` (0.890 ROC-AUC)
**Trust Status**: ✅ VALIDATED (2026-01-12)
  - Trust Score: 0.95
  - Pass Rate: 90.9% (10/11 tests)
  - Final Recommendation: ACCEPT
  - CV Stability: mean=0.8836, std=0.0305
  - Inference: ~5.2ms/molecule (well under 100ms limit)
  - Baseline Comparison: +8.97% improvement over baseline
**Previous Champion**: `gen9b.py` (0.886 ROC-AUC, validated)
**Last Session**: Gen11-14 evolution completed. Gen14a achieved 0.892 but FAILED trust validation (edge case handling issues with MLP). Gen12c promoted to champion.

## Validation Commands

```bash
# Basic evaluation (NOT sufficient alone)
./venv/bin/python evaluate.py <solution.py> --json

# Full trust validation (REQUIRED for champions)
./venv/bin/python escalated_validation.py <solution.py>

# Run evolution properly (uses trust system)
./venv/bin/python -m evolve_sdk --config=evolve_config.json --mode=ml
```

## Bypass Trust (ONLY if explicitly requested)

If the user explicitly says "skip trust validation" or "bypass trust", you may skip it.
Otherwise, ALWAYS run trust validation for any champion candidate.
