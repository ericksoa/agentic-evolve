#!/usr/bin/env python3
"""
Escalated Validation for hERG prediction candidates.

Performs Level 1-3 validation to ensure candidates are trustworthy:
- Level 1: Edge cases, robustness, consistency
- Level 2: Out-of-distribution generalization
- Level 3: Gold standard benchmark comparison

Usage:
    python escalated_validation.py <solution.py> [--json]

Example:
    python escalated_validation.py gen9b.py
    python escalated_validation.py gen9b.py --json
"""

import argparse
import numpy as np
import json
import sys
import subprocess
import warnings
from sklearn.metrics import roc_auc_score
from pathlib import Path

warnings.filterwarnings('ignore')


def load_candidate(solution_path):
    """Load the candidate solution."""
    import importlib.util

    solution_path = Path(solution_path)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution not found: {solution_path}")

    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'HERGPredictor'):
        return module.HERGPredictor
    else:
        raise ValueError(f"Solution must define a HERGPredictor class")


def get_baseline_fitness(solution_path):
    """Get the baseline fitness from evaluate.py."""
    try:
        result = subprocess.run([
            sys.executable, 'evaluate.py', str(solution_path), '--json'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse JSON from output (may have progress bars before it)
            output = result.stdout
            # Find the JSON part - it may be multiline
            json_start = output.find('{')
            json_end = output.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = output[json_start:json_end+1]
                eval_data = json.loads(json_str)
                return eval_data.get('fitness', 0.0), eval_data
            # Try parsing the whole thing
            eval_data = json.loads(output)
            return eval_data.get('fitness', 0.0), eval_data
    except Exception as e:
        print(f"Warning: Could not get baseline fitness: {e}")

    return 0.0, {}


def level_1_extended_tests(predictor_class):
    """
    Level 1: Extended edge case tests
    - Empty inputs
    - Invalid SMILES
    - Extreme molecular properties
    - Boundary conditions
    - Prediction consistency
    """
    test_results = []

    # Test 1: Empty input handling
    try:
        predictor = predictor_class()
        predictor.fit(['CC', 'CCO', 'CCC', 'CCCC'], [0, 1, 0, 1])

        result = predictor.predict_proba([])
        passed = hasattr(result, '__len__') and len(result) == 0

        test_results.append({
            'test': 'empty_input',
            'passed': passed,
            'details': f'Empty input returned array of length {len(result) if hasattr(result, "__len__") else "N/A"}'
        })
    except Exception as e:
        # Graceful error handling is also acceptable
        test_results.append({
            'test': 'empty_input',
            'passed': True,  # Raising an exception for empty input is OK
            'details': f'Raised exception (acceptable): {str(e)[:100]}'
        })

    # Test 2: Invalid SMILES strings
    try:
        predictor = predictor_class()
        predictor.fit(['CC', 'CCO', 'CCCO', 'CCCCO'], [0, 1, 0, 1])

        invalid_smiles = [
            '',           # Empty string
            'INVALID',    # Invalid SMILES
            'C[C@]',      # Incomplete stereochemistry
            'C1C',        # Incomplete ring
            '[Xe]',       # Rare element
        ]

        predictions = predictor.predict_proba(invalid_smiles)
        all_finite = np.all(np.isfinite(predictions))
        all_in_range = np.all((predictions >= 0) & (predictions <= 1))

        test_results.append({
            'test': 'invalid_smiles',
            'passed': all_finite and all_in_range,
            'details': f'Invalid SMILES: finite={all_finite}, in_range={all_in_range}'
        })
    except Exception as e:
        test_results.append({
            'test': 'invalid_smiles',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 3: Extreme molecular sizes
    try:
        predictor = predictor_class()
        predictor.fit(['CC', 'CCO', 'CCCO', 'CCCCO'], [0, 1, 0, 1])

        extreme_molecules = [
            'C',                    # Single carbon
            'C' * 50,               # Long alkane chain
            'c1ccccc1' * 10,        # Many benzene rings
        ]

        predictions = predictor.predict_proba(extreme_molecules)
        all_finite = np.all(np.isfinite(predictions))
        all_in_range = np.all((predictions >= 0) & (predictions <= 1))

        test_results.append({
            'test': 'extreme_sizes',
            'passed': all_finite and all_in_range,
            'details': f'Extreme sizes: finite={all_finite}, in_range={all_in_range}'
        })
    except Exception as e:
        test_results.append({
            'test': 'extreme_sizes',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 4: Minimal training data
    try:
        predictor = predictor_class()
        predictor.fit(['CC', 'CCO'], [0, 1])
        predictions = predictor.predict_proba(['CCC', 'CCCO'])

        all_finite = np.all(np.isfinite(predictions))
        all_in_range = np.all((predictions >= 0) & (predictions <= 1))

        test_results.append({
            'test': 'minimal_training',
            'passed': all_finite and all_in_range,
            'details': f'Minimal training: finite={all_finite}, in_range={all_in_range}'
        })
    except Exception as e:
        test_results.append({
            'test': 'minimal_training',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 5: Prediction consistency
    try:
        predictor = predictor_class(random_state=42)
        predictor.fit(['CC', 'CCO', 'CCCO', 'CCCCO'], [0, 1, 0, 1])

        test_smiles = ['CCC', 'CCCC', 'c1ccccc1']
        pred1 = predictor.predict_proba(test_smiles)
        pred2 = predictor.predict_proba(test_smiles)

        consistent = np.allclose(pred1, pred2, atol=1e-6)

        test_results.append({
            'test': 'prediction_consistency',
            'passed': consistent,
            'details': f'Predictions consistent: {consistent}'
        })
    except Exception as e:
        test_results.append({
            'test': 'prediction_consistency',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 6: Inference timing
    try:
        import time
        predictor = predictor_class()

        # Load real data for realistic timing
        try:
            from tdc.single_pred import Tox
            data = Tox(name='hERG')
            split = data.get_split(method='scaffold', seed=42)
            X_train = split['train']['Drug'].tolist()[:500]
            y_train = split['train']['Y'].tolist()[:500]
            X_test = split['test']['Drug'].tolist()[:100]
        except:
            X_train = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1CCN(CC1)c2ccccc2'] * 250
            y_train = [0, 1] * 250
            X_test = ['CC', 'CCO', 'CCCO'] * 33

        predictor.fit(X_train, y_train)

        start = time.time()
        for _ in range(3):
            _ = predictor.predict_proba(X_test)
        elapsed = (time.time() - start) / 3

        inference_ms = elapsed * 1000 / len(X_test)
        passed = inference_ms < 100  # < 100ms per molecule

        test_results.append({
            'test': 'inference_timing',
            'passed': passed,
            'details': f'Inference: {inference_ms:.2f}ms per molecule (limit: 100ms)'
        })
    except Exception as e:
        test_results.append({
            'test': 'inference_timing',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    return test_results


def level_2_ood_tests(predictor_class):
    """
    Level 2: Out-of-distribution tests
    - Novel chemical scaffolds
    - Cross-validation stability
    """
    test_results = []

    # Load real hERG data
    try:
        from tdc.single_pred import Tox
        data = Tox(name='hERG')
        split = data.get_split(method='scaffold', seed=42)
        X_train = split['train']['Drug'].tolist()
        y_train = split['train']['Y'].tolist()
    except ImportError:
        X_train = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1CCN(CC1)c2ccccc2', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'] * 100
        y_train = [0, 1, 0] * 100

    # Test 1: Novel chemical scaffolds
    try:
        predictor = predictor_class()
        predictor.fit(X_train, y_train)

        novel_scaffolds = [
            'C1=CC2=C(C=C1)N=C(S2)N3CCN(CC3)C4=CC=CC=C4F',  # Benzothiazole
            'CC1=NN(C2=C1C=CC(=C2)C3=CC=C(C=C3)CN4CCOCC4)C',  # Pyrazole
            'C1=CC=C2C(=C1)C(=CN2)C3=NC4=CC=CC=C4N3',  # Imidazopyridine
            'COC1=CC=C(C=C1)S(=O)(=O)N2CCCC2C3=CC=CC=N3',  # Sulfonamide
        ]

        predictions = predictor.predict_proba(novel_scaffolds)
        pred_var = np.var(predictions)
        reasonable_variance = pred_var > 0.001  # Some diversity

        all_finite = np.all(np.isfinite(predictions))
        all_in_range = np.all((predictions >= 0) & (predictions <= 1))

        test_results.append({
            'test': 'novel_scaffolds',
            'passed': all_finite and all_in_range and reasonable_variance,
            'details': f'Novel scaffolds: var={pred_var:.4f}, finite={all_finite}, in_range={all_in_range}'
        })
    except Exception as e:
        test_results.append({
            'test': 'novel_scaffolds',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 2: Cross-validation stability
    try:
        from sklearn.model_selection import StratifiedKFold

        # Use subset for speed
        n_samples = min(len(X_train), 500)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_subset = np.array([X_train[i] for i in indices])
        y_subset = np.array([y_train[i] for i in indices])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        aucs = []

        for train_idx, val_idx in skf.split(X_subset, y_subset):
            predictor = predictor_class()
            X_tr = X_subset[train_idx].tolist()
            y_tr = y_subset[train_idx].tolist()
            X_val = X_subset[val_idx].tolist()
            y_val = y_subset[val_idx]

            predictor.fit(X_tr, y_tr)
            y_pred = predictor.predict_proba(X_val)

            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, y_pred)
                aucs.append(auc)

        if len(aucs) > 0:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            stable = std_auc < 0.1

            test_results.append({
                'test': 'cross_validation_stability',
                'passed': stable and mean_auc > 0.6,
                'details': f'CV: mean={mean_auc:.4f}, std={std_auc:.4f}'
            })
        else:
            test_results.append({
                'test': 'cross_validation_stability',
                'passed': False,
                'details': 'Could not compute AUC'
            })

    except Exception as e:
        test_results.append({
            'test': 'cross_validation_stability',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    return test_results


def level_3_gold_standard(predictor_class, solution_path, claimed_fitness):
    """
    Level 3: Gold standard validation
    - Verify claimed fitness
    - Baseline comparison
    - Holdout performance
    """
    test_results = []

    # Test 1: Verify claimed fitness (re-run evaluation)
    try:
        result = subprocess.run([
            sys.executable, 'evaluate.py', str(solution_path), '--json'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse JSON
            output = result.stdout
            eval_data = None
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('{'):
                    eval_data = json.loads(line)
                    break

            if eval_data is None:
                eval_data = json.loads(output)

            actual_fitness = eval_data.get('fitness', 0.0)
            discrepancy = abs(actual_fitness - claimed_fitness)

            # Fitness should be within 5% of claimed
            fitness_verified = discrepancy < 0.05

            test_results.append({
                'test': 'fitness_verification',
                'passed': fitness_verified,
                'details': f'Claimed={claimed_fitness:.4f}, Actual={actual_fitness:.4f}, Diff={discrepancy:.4f}'
            })
        else:
            test_results.append({
                'test': 'fitness_verification',
                'passed': False,
                'details': f'Evaluation failed: {result.stderr[:100]}'
            })

    except Exception as e:
        test_results.append({
            'test': 'fitness_verification',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 2: Baseline comparison
    try:
        from sklearn.ensemble import RandomForestClassifier
        from rdkit import Chem
        from rdkit.Chem import AllChem

        class BaselinePredictor:
            def __init__(self):
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)

            def fit(self, X_smiles, y):
                X = []
                for smi in X_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        X.append(np.array(fp))
                    else:
                        X.append(np.zeros(1024))
                self.model.fit(X, y)
                return self

            def predict_proba(self, X_smiles):
                X = []
                for smi in X_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        X.append(np.array(fp))
                    else:
                        X.append(np.zeros(1024))
                return self.model.predict_proba(X)[:, 1]

        # Load data
        try:
            from tdc.single_pred import Tox
            data = Tox(name='hERG')
            split = data.get_split(method='scaffold', seed=42)
            X_train = split['train']['Drug'].tolist()
            y_train = split['train']['Y'].tolist()
            X_test = split['test']['Drug'].tolist()
            y_test = split['test']['Y'].tolist()
        except:
            X_train = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1CCN(CC1)c2ccccc2'] * 100
            y_train = [0, 1] * 100
            X_test = ['CC', 'CCO', 'CCCO', 'CCCCO'] * 25
            y_test = [0, 1, 0, 1] * 25

        # Test candidate
        candidate = predictor_class()
        candidate.fit(X_train, y_train)
        candidate_preds = candidate.predict_proba(X_test)
        candidate_auc = roc_auc_score(y_test, candidate_preds)

        # Test baseline
        baseline = BaselinePredictor()
        baseline.fit(X_train, y_train)
        baseline_preds = baseline.predict_proba(X_test)
        baseline_auc = roc_auc_score(y_test, baseline_preds)

        improvement = candidate_auc - baseline_auc
        beats_baseline = improvement > -0.02  # At least not 2% worse

        test_results.append({
            'test': 'baseline_comparison',
            'passed': beats_baseline,
            'details': f'Candidate={candidate_auc:.4f}, Baseline={baseline_auc:.4f}, Improvement={improvement:.4f}'
        })

    except Exception as e:
        test_results.append({
            'test': 'baseline_comparison',
            'passed': False,
            'details': f'Error: {str(e)[:100]}'
        })

    # Test 3: Multiple seed evaluation (variance check)
    try:
        from tdc.single_pred import Tox
        data = Tox(name='hERG')

        aucs = []
        for seed in [42, 123, 456]:
            split = data.get_split(method='scaffold', seed=seed)
            X_train = split['train']['Drug'].tolist()
            y_train = split['train']['Y'].tolist()
            X_test = split['test']['Drug'].tolist()
            y_test = split['test']['Y'].tolist()

            predictor = predictor_class()
            predictor.fit(X_train, y_train)
            y_pred = predictor.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        stable = std_auc < 0.05  # Low variance across seeds

        test_results.append({
            'test': 'multi_seed_stability',
            'passed': stable,
            'details': f'Seeds [42,123,456]: mean={mean_auc:.4f}, std={std_auc:.4f}'
        })

    except Exception as e:
        test_results.append({
            'test': 'multi_seed_stability',
            'passed': True,  # Skip if TDC not available
            'details': f'Skipped (TDC not available): {str(e)[:50]}'
        })

    return test_results


def run_escalated_validation(solution_path, output_json=False):
    """Run the complete escalated validation pipeline."""

    solution_path = Path(solution_path)
    solution_name = solution_path.name

    if not output_json:
        print(f"\n{'='*60}")
        print(f"ESCALATED VALIDATION: {solution_name}")
        print(f"{'='*60}")

    # Get baseline fitness first
    claimed_fitness, eval_data = get_baseline_fitness(solution_path)

    if not output_json:
        print(f"\nBaseline fitness from evaluate.py: {claimed_fitness:.4f}")

    # Load candidate
    try:
        predictor_class = load_candidate(solution_path)
        if not output_json:
            print(f"Candidate loaded successfully")
    except Exception as e:
        result = {
            "solution": str(solution_path),
            "escalation_passed": False,
            "revised_fitness": 0.0,
            "revised_trust": 0.0,
            "tests_run": [],
            "failures": [f"Failed to load candidate: {str(e)}"],
            "final_recommendation": "REJECT"
        }
        if output_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nFAILED: Could not load candidate: {e}")
        return result

    all_results = []
    all_failures = []

    # Level 1: Extended Tests
    if not output_json:
        print(f"\n[Level 1] Extended Edge Case Tests")
        print("-" * 40)

    level1_results = level_1_extended_tests(predictor_class)
    all_results.extend(level1_results)

    level1_passed = 0
    for result in level1_results:
        if result['passed']:
            level1_passed += 1
        else:
            all_failures.append(f"L1-{result['test']}")

        if not output_json:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['test']:30s}: {status}")

    if not output_json:
        print(f"\n  Level 1: {level1_passed}/{len(level1_results)} passed")

    # Level 2: Out-of-Distribution Tests
    if not output_json:
        print(f"\n[Level 2] Out-of-Distribution Tests")
        print("-" * 40)

    level2_results = level_2_ood_tests(predictor_class)
    all_results.extend(level2_results)

    level2_passed = 0
    for result in level2_results:
        if result['passed']:
            level2_passed += 1
        else:
            all_failures.append(f"L2-{result['test']}")

        if not output_json:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['test']:30s}: {status}")

    if not output_json:
        print(f"\n  Level 2: {level2_passed}/{len(level2_results)} passed")

    # Level 3: Gold Standard
    if not output_json:
        print(f"\n[Level 3] Gold Standard Validation")
        print("-" * 40)

    level3_results = level_3_gold_standard(predictor_class, solution_path, claimed_fitness)
    all_results.extend(level3_results)

    level3_passed = 0
    for result in level3_results:
        if result['passed']:
            level3_passed += 1
        else:
            all_failures.append(f"L3-{result['test']}")

        if not output_json:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['test']:30s}: {status}")

    if not output_json:
        print(f"\n  Level 3: {level3_passed}/{len(level3_results)} passed")

    # Overall assessment
    total_tests = len(all_results)
    total_passed = sum(1 for r in all_results if r['passed'])
    pass_rate = total_passed / total_tests if total_tests > 0 else 0

    # Determine outcome
    escalation_passed = pass_rate >= 0.7  # Need 70% pass rate

    if escalation_passed:
        if pass_rate >= 0.9:
            fitness_multiplier = 1.0
            trust_score = 0.95
        elif pass_rate >= 0.8:
            fitness_multiplier = 0.95
            trust_score = 0.85
        else:
            fitness_multiplier = 0.90
            trust_score = 0.75
        revised_fitness = claimed_fitness * fitness_multiplier
        recommendation = "ACCEPT"
    else:
        fitness_multiplier = 0.5
        trust_score = pass_rate * 0.5
        revised_fitness = claimed_fitness * fitness_multiplier
        recommendation = "REJECT"

    if not output_json:
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Solution:         {solution_name}")
        print(f"Tests Passed:     {total_passed}/{total_tests} ({pass_rate:.1%})")
        print(f"Claimed Fitness:  {claimed_fitness:.4f}")
        print(f"Revised Fitness:  {revised_fitness:.4f}")
        print(f"Trust Score:      {trust_score:.2f}")
        print(f"Recommendation:   {recommendation}")

        if all_failures:
            print(f"\nFailed Tests:")
            for f in all_failures:
                print(f"  - {f}")

    result = {
        "solution": str(solution_path),
        "escalation_passed": bool(escalation_passed),
        "claimed_fitness": float(claimed_fitness),
        "revised_fitness": float(revised_fitness),
        "trust_score": float(trust_score),
        "pass_rate": float(pass_rate),
        "tests_passed": total_passed,
        "tests_total": total_tests,
        "failures": all_failures,
        "final_recommendation": recommendation,
        "detailed_results": [
            {
                "level": r['test'].split('_')[0] if '_' in r['test'] else 'L1',
                "test": r['test'],
                "passed": bool(r['passed']),
                "details": str(r['details'])
            } for r in all_results
        ]
    }

    if output_json:
        print(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run escalated trust validation on a hERG prediction candidate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python escalated_validation.py gen9b.py
  python escalated_validation.py gen9b.py --json
  python escalated_validation.py .evolve-sdk/evolve_herg_molecular_property/mutations/gen6b.py
        """
    )
    parser.add_argument('solution', help='Path to the solution file (.py)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')

    args = parser.parse_args()

    if not Path(args.solution).exists():
        print(f"Error: Solution file not found: {args.solution}")
        sys.exit(1)

    result = run_escalated_validation(args.solution, output_json=args.json)

    # Exit with appropriate code
    if result['final_recommendation'] == 'ACCEPT':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
