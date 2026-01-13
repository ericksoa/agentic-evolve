#!/usr/bin/env python3
"""
Adversarial validation to detect potential evaluator exploitation.
Tests for overfitting, hardcoding, and other forms of gaming the evaluation.
"""

import numpy as np
import json
import sys
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Import both candidate and baseline
sys.path.insert(0, '.evolve-sdk/evolve_molecular_property_pred/mutations')
from gen4c import HERGPredictor as CandidatePredictor
sys.path.insert(0, '.')
from baseline_descriptors import HERGPredictor as BaselinePredictor

warnings.filterwarnings('ignore')


class AdversaryValidator:
    """Adversarial validator to detect evaluator exploitation."""

    def __init__(self):
        self.flags = []
        self.analysis_results = {}

    def flag_concern(self, flag_type, severity, description):
        """Flag a potential concern."""
        self.flags.append({
            'type': flag_type,
            'severity': severity,  # 'low', 'medium', 'high'
            'description': description
        })
        print(f"🚩 FLAG ({severity.upper()}): {flag_type} - {description}")

    def load_tdc_data(self):
        """Load the actual TDC hERG data."""
        try:
            from tdc.single_pred import Tox
            data = Tox(name='hERG')
            split = data.get_split(method='scaffold', seed=42)
            return {
                'train': (split['train']['Drug'].tolist(), split['train']['Y'].tolist()),
                'valid': (split['valid']['Drug'].tolist(), split['valid']['Y'].tolist()),
                'test': (split['test']['Drug'].tolist(), split['test']['Y'].tolist())
            }
        except ImportError:
            print("Warning: PyTDC not available, using synthetic data")
            # Create synthetic data that mimics real patterns
            synthetic_smiles = [
                'CCO', 'CCC', 'CCCC', 'c1ccccc1', 'CCN(CC)CC', 'CC(=O)C',
                'c1ccc(O)cc1', 'c1ccc(N)cc1', 'CCOC(=O)C', 'CC(C)C',
            ] * 100
            synthetic_labels = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0] * 100

            # Shuffle and split
            indices = list(range(len(synthetic_smiles)))
            np.random.RandomState(42).shuffle(indices)

            n_train = int(0.7 * len(indices))
            n_valid = int(0.15 * len(indices))

            train_idx = indices[:n_train]
            valid_idx = indices[n_train:n_train+n_valid]
            test_idx = indices[n_train+n_valid:]

            return {
                'train': ([synthetic_smiles[i] for i in train_idx], [synthetic_labels[i] for i in train_idx]),
                'valid': ([synthetic_smiles[i] for i in valid_idx], [synthetic_labels[i] for i in valid_idx]),
                'test': ([synthetic_smiles[i] for i in test_idx], [synthetic_labels[i] for i in test_idx])
            }

    def test_model_memorization(self, predictor_class, train_smiles, train_labels):
        """Test if model just memorizes training data."""
        print("Testing for memorization...")

        predictor = predictor_class(random_state=42)
        predictor.fit(train_smiles, train_labels)

        # Test on training data
        train_proba = predictor.predict_proba(train_smiles)
        train_auc = roc_auc_score(train_labels, train_proba)

        # Test on modified training data (small changes)
        modified_smiles = []
        for smi in train_smiles[:50]:  # Test subset for speed
            if len(smi) > 2:
                # Make small modification - add a carbon
                if 'C' in smi:
                    modified_smiles.append(smi + 'C')
                else:
                    modified_smiles.append('C' + smi)
            else:
                modified_smiles.append(smi + 'C')

        modified_proba = predictor.predict_proba(modified_smiles)

        # Check if predictions are suspiciously similar to original training
        if train_auc > 0.99:
            self.flag_concern(
                "perfect_training_fit",
                "high",
                f"Perfect/near-perfect training AUC: {train_auc:.4f}"
            )

        self.analysis_results['memorization'] = {
            'train_auc': train_auc,
            'modified_predictions_variance': np.var(modified_proba)
        }

    def test_architecture_complexity(self, predictor):
        """Test if architecture seems reasonable for the problem."""
        print("Testing architecture complexity...")

        if hasattr(predictor, 'get_architecture_info'):
            arch_info = predictor.get_architecture_info()
            total_features = arch_info.get('total_features', 0)
            hidden_layers = arch_info.get('hidden_layers', ())

            # Calculate approximate parameter count
            if isinstance(hidden_layers, tuple) and len(hidden_layers) > 0:
                param_count = total_features * hidden_layers[0]
                for i in range(1, len(hidden_layers)):
                    param_count += hidden_layers[i-1] * hidden_layers[i]
                param_count += hidden_layers[-1] * 2  # Output layer

                self.analysis_results['architecture'] = {
                    'total_features': total_features,
                    'hidden_layers': hidden_layers,
                    'estimated_parameters': param_count
                }

                # Flag if architecture seems overly complex
                if param_count > 1000000:  # > 1M parameters
                    self.flag_concern(
                        "overly_complex_architecture",
                        "medium",
                        f"Very large model: ~{param_count:,} parameters"
                    )

    def test_cross_dataset_generalization(self, predictor_class):
        """Test generalization to slightly different data distributions."""
        print("Testing cross-dataset generalization...")

        data = self.load_tdc_data()
        train_smiles, train_labels = data['train']
        test_smiles, test_labels = data['test']

        # Train on original data
        predictor = predictor_class(random_state=42)
        predictor.fit(train_smiles, train_labels)
        original_test_auc = roc_auc_score(test_labels, predictor.predict_proba(test_smiles))

        # Create "shifted" test set - filter to specific molecular weight ranges
        shifted_smiles = []
        shifted_labels = []

        for smi, label in zip(test_smiles, test_labels):
            # Simple heuristic: longer SMILES = higher MW
            if len(smi) > 10:  # Focus on larger molecules
                shifted_smiles.append(smi)
                shifted_labels.append(label)

        if len(shifted_labels) > 10:  # Need enough samples for AUC
            if len(set(shifted_labels)) > 1:  # Need both classes
                shifted_test_auc = roc_auc_score(shifted_labels, predictor.predict_proba(shifted_smiles))

                performance_drop = original_test_auc - shifted_test_auc

                if performance_drop > 0.2:
                    self.flag_concern(
                        "poor_generalization",
                        "high",
                        f"Large performance drop on shifted data: {performance_drop:.3f}"
                    )

                self.analysis_results['generalization'] = {
                    'original_test_auc': original_test_auc,
                    'shifted_test_auc': shifted_test_auc,
                    'performance_drop': performance_drop,
                    'shifted_test_size': len(shifted_labels)
                }

    def test_feature_importance_sanity(self, predictor):
        """Test if feature usage makes chemical sense."""
        print("Testing feature importance sanity...")

        # For neural networks, we can't easily get feature importance
        # But we can test if the predictor is using reasonable molecular descriptors

        # Test with molecules that should have very different hERG risk
        low_risk_smiles = [
            'O',  # Water
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
        ]

        high_risk_smiles = [
            'CCN(CC)CCC1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3)O',  # Diphenhydramine-like
            'CN1CCC[C@H]1C2=CN=CC=C2',  # Nicotine-like
        ]

        predictor_test = type(predictor)(random_state=42)

        # Train on small dataset
        train_smiles = ['CCO', 'CCC', 'c1ccccc1', 'CCN(CC)CC'] * 20
        train_labels = [0, 0, 1, 1] * 20
        predictor_test.fit(train_smiles, train_labels)

        low_risk_proba = np.mean(predictor_test.predict_proba(low_risk_smiles))
        high_risk_proba = np.mean(predictor_test.predict_proba(high_risk_smiles))

        if low_risk_proba > high_risk_proba:
            self.flag_concern(
                "inverted_chemical_logic",
                "medium",
                f"Low-risk molecules scored higher than high-risk: {low_risk_proba:.3f} vs {high_risk_proba:.3f}"
            )

        self.analysis_results['chemical_sanity'] = {
            'low_risk_avg_proba': low_risk_proba,
            'high_risk_avg_proba': high_risk_proba,
            'chemical_logic_correct': low_risk_proba < high_risk_proba
        }

    def compare_with_baseline(self, candidate_class, baseline_class):
        """Compare candidate performance with baseline."""
        print("Comparing with baseline...")

        data = self.load_tdc_data()
        train_smiles, train_labels = data['train']
        test_smiles, test_labels = data['test']

        # Test candidate
        candidate = candidate_class(random_state=42)
        candidate.fit(train_smiles, train_labels)
        candidate_auc = roc_auc_score(test_labels, candidate.predict_proba(test_smiles))

        # Test baseline
        baseline = baseline_class(random_state=42)
        baseline.fit(train_smiles, train_labels)
        baseline_auc = roc_auc_score(test_labels, baseline.predict_proba(test_smiles))

        improvement = candidate_auc - baseline_auc

        # Flag if improvement is suspiciously large
        if improvement > 0.15:  # More than 15% absolute improvement
            self.flag_concern(
                "suspiciously_large_improvement",
                "high",
                f"Very large improvement over baseline: {improvement:.3f} ({candidate_auc:.3f} vs {baseline_auc:.3f})"
            )

        # Flag if improvement is negative (performing worse than baseline)
        if improvement < -0.05:
            self.flag_concern(
                "worse_than_baseline",
                "medium",
                f"Performing worse than baseline: {improvement:.3f}"
            )

        self.analysis_results['baseline_comparison'] = {
            'candidate_auc': candidate_auc,
            'baseline_auc': baseline_auc,
            'improvement': improvement
        }

        return candidate_auc, baseline_auc

    def run_adversarial_validation(self):
        """Run complete adversarial validation."""
        print("=" * 60)
        print("ADVERSARIAL VALIDATION")
        print("=" * 60)

        data = self.load_tdc_data()
        train_smiles, train_labels = data['train']

        # Test candidate architecture
        candidate = CandidatePredictor(random_state=42)
        self.test_architecture_complexity(candidate)

        # Test memorization
        self.test_model_memorization(CandidatePredictor, train_smiles, train_labels)

        # Test generalization
        self.test_cross_dataset_generalization(CandidatePredictor)

        # Test chemical sanity
        self.test_feature_importance_sanity(candidate)

        # Compare with baseline
        candidate_auc, baseline_auc = self.compare_with_baseline(CandidatePredictor, BaselinePredictor)

        print("\n" + "=" * 60)
        print("ADVERSARIAL VALIDATION RESULTS")
        print("=" * 60)

        print(f"\nPerformance:")
        print(f"  Candidate AUC: {candidate_auc:.4f}")
        print(f"  Baseline AUC:  {baseline_auc:.4f}")
        print(f"  Improvement:   {candidate_auc - baseline_auc:.4f}")

        print(f"\nFlags raised: {len(self.flags)}")
        if self.flags:
            for flag in self.flags:
                print(f"  🚩 {flag['severity'].upper()}: {flag['type']} - {flag['description']}")
        else:
            print("  ✅ No major concerns detected")

        # Calculate trust score
        high_severity_flags = sum(1 for f in self.flags if f['severity'] == 'high')
        medium_severity_flags = sum(1 for f in self.flags if f['severity'] == 'medium')

        trust_score = 1.0 - (high_severity_flags * 0.3 + medium_severity_flags * 0.1)
        trust_score = max(0.0, trust_score)

        # Determine recommendation
        if trust_score >= 0.7 and candidate_auc > baseline_auc:
            recommendation = "accept"
        elif high_severity_flags > 0:
            recommendation = "reject"
        else:
            recommendation = "challenge"

        # Calculate adjusted fitness (penalize for suspicious behavior)
        fitness_penalty = high_severity_flags * 0.1 + medium_severity_flags * 0.05
        adjusted_fitness = max(0.0, candidate_auc - fitness_penalty)

        return {
            'candidate_auc': candidate_auc,
            'baseline_auc': baseline_auc,
            'trust_score': trust_score,
            'flags': self.flags,
            'recommendation': recommendation,
            'adjusted_fitness': adjusted_fitness,
            'analysis_details': self.analysis_results
        }


if __name__ == "__main__":
    validator = AdversaryValidator()
    results = validator.run_adversarial_validation()

    print(f"\nFinal Assessment:")
    print(f"  Trust Score: {results['trust_score']:.2f}")
    print(f"  Recommendation: {results['recommendation'].upper()}")
    print(f"  Adjusted Fitness: {results['adjusted_fitness']:.4f}")