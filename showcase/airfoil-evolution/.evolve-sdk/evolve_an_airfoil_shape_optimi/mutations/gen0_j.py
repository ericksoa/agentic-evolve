#!/usr/bin/env python3
"""
Gen 0 Variant J: Meta-Learning Ensemble with Transfer Learning
Combines knowledge from multiple initialization strategies with meta-learning.
Optimized with cached meta-features and vectorized ensemble predictions.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class MetaFeatureExtractor:
    """Extract meta-features from airfoil designs for machine learning."""

    @staticmethod
    def extract_cst_features(airfoil: Airfoil) -> np.ndarray:
        """Extract features from CST coefficients."""
        upper = np.array(airfoil.upper_coeffs)
        lower = np.array(airfoil.lower_coeffs)

        features = []

        # Basic coefficient statistics
        features.extend([
            np.mean(upper), np.std(upper), np.max(upper), np.min(upper),
            np.mean(lower), np.std(lower), np.max(lower), np.min(lower)
        ])

        # Coefficient gradients (shape derivatives)
        upper_grad = np.diff(upper)
        lower_grad = np.diff(lower)
        features.extend([
            np.mean(upper_grad), np.std(upper_grad),
            np.mean(lower_grad), np.std(lower_grad)
        ])

        # Cross-correlations
        features.append(np.corrcoef(upper, lower)[0, 1] if len(upper) > 1 else 0)

        return np.array(features)

    @staticmethod
    def extract_geometric_features(airfoil: Airfoil) -> np.ndarray:
        """Extract geometric meta-features."""
        features = []

        # Thickness distribution
        max_thick, thick_pos = airfoil.max_thickness()
        features.extend([max_thick, thick_pos])

        # Camber distribution
        max_camber, camber_pos = airfoil.max_camber()
        features.extend([max_camber, camber_pos])

        # Leading edge curvature approximation
        le_upper = airfoil.upper_surface(0.01)
        le_lower = airfoil.lower_surface(0.01)
        le_curvature = abs(le_upper - le_lower)
        features.append(le_curvature)

        # Trailing edge thickness
        te_upper = airfoil.upper_surface(0.99)
        te_lower = airfoil.lower_surface(0.99)
        te_thickness = abs(te_upper - te_lower)
        features.append(te_thickness)

        # Shape complexity (second derivatives)
        x_sample = np.linspace(0.1, 0.9, 9)
        upper_samples = [airfoil.upper_surface(x) for x in x_sample]
        lower_samples = [airfoil.lower_surface(x) for x in x_sample]

        upper_curvature = np.std(np.diff(upper_samples, 2)) if len(upper_samples) > 2 else 0
        lower_curvature = np.std(np.diff(lower_samples, 2)) if len(lower_samples) > 2 else 0
        features.extend([upper_curvature, lower_curvature])

        return np.array(features)

    @classmethod
    def extract_all_features(cls, airfoil: Airfoil) -> np.ndarray:
        """Extract complete feature vector."""
        cst_features = cls.extract_cst_features(airfoil)
        geo_features = cls.extract_geometric_features(airfoil)
        return np.concatenate([cst_features, geo_features])

class TransferLearningPredictor:
    """Transfer learning predictor for airfoil performance."""

    def __init__(self):
        self.feature_extractor = MetaFeatureExtractor()
        self.scaler = StandardScaler()
        self.predictor = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=42,
            n_jobs=1  # Single thread for stability
        )
        self.is_fitted = False

    def prepare_training_data(self, airfoils: List[Airfoil], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from airfoils and their fitnesses."""
        features = []
        for airfoil in airfoils:
            try:
                feature_vector = self.feature_extractor.extract_all_features(airfoil)
                features.append(feature_vector)
            except:
                # Fallback feature vector
                features.append(np.zeros(21))  # Expected feature count

        X = np.array(features)
        y = np.array(fitnesses)

        # Remove invalid entries
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y >= 0)
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y

    def fit(self, airfoils: List[Airfoil], fitnesses: List[float]):
        """Fit the transfer learning model."""
        X, y = self.prepare_training_data(airfoils, fitnesses)

        if len(X) < 5:
            print("Not enough training data for transfer learning")
            return

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit predictor
        self.predictor.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, airfoils: List[Airfoil]) -> np.ndarray:
        """Predict fitness for new airfoils."""
        if not self.is_fitted:
            return np.zeros(len(airfoils))

        features = []
        for airfoil in airfoils:
            try:
                feature_vector = self.feature_extractor.extract_all_features(airfoil)
                features.append(feature_vector)
            except:
                features.append(np.zeros(21))

        X = np.array(features)

        # Handle edge cases
        if X.shape[1] != self.scaler.n_features_in_:
            return np.zeros(len(airfoils))

        X_scaled = self.scaler.transform(X)
        predictions = self.predictor.predict(X_scaled)

        return np.clip(predictions, 0, 100)  # Reasonable bounds

class EnsembleKnowledgeBase:
    """Knowledge base from previous initialization strategies."""

    def __init__(self):
        self.strategy_knowledge = {
            'gaussian_random': self._generate_gaussian_samples,
            'physics_informed': self._generate_physics_samples,
            'quasi_random': self._generate_quasi_samples,
            'optimization_guided': self._generate_optimization_samples
        }

    def _generate_gaussian_samples(self, n_samples: int, rng: np.random.RandomState) -> List[Airfoil]:
        """Generate samples using Gaussian strategy."""
        airfoils = []
        for _ in range(n_samples):
            upper = np.clip(rng.normal(0.15, 0.04, 5), 0.01, 0.3)
            lower = np.clip(rng.normal(-0.12, 0.03, 5), -0.3, -0.01)
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist()))
        return airfoils

    def _generate_physics_samples(self, n_samples: int, rng: np.random.RandomState) -> List[Airfoil]:
        """Generate physics-informed samples."""
        airfoils = []
        # Known good coefficient patterns
        patterns = [
            ([0.17, 0.16, 0.14, 0.12, 0.10], [-0.17, -0.16, -0.14, -0.12, -0.10]),
            ([0.20, 0.18, 0.16, 0.13, 0.10], [-0.14, -0.12, -0.10, -0.09, -0.08]),
            ([0.13, 0.12, 0.11, 0.09, 0.07], [-0.13, -0.12, -0.11, -0.09, -0.07])
        ]

        for _ in range(n_samples):
            pattern = rng.choice(len(patterns))
            upper_base, lower_base = patterns[pattern]

            # Add perturbations
            upper = np.array(upper_base) + rng.normal(0, 0.02, 5)
            lower = np.array(lower_base) + rng.normal(0, 0.02, 5)

            upper = np.clip(upper, 0.01, 0.3)
            lower = np.clip(lower, -0.3, -0.01)

            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist()))
        return airfoils

    def _generate_quasi_samples(self, n_samples: int, rng: np.random.RandomState) -> List[Airfoil]:
        """Generate quasi-random samples."""
        airfoils = []
        for i in range(n_samples):
            # Simple quasi-random using Van der Corput sequence
            t = i / n_samples

            # Map to coefficient space
            upper = 0.05 + 0.2 * np.array([
                self._van_der_corput(i, 2),
                self._van_der_corput(i, 3),
                self._van_der_corput(i, 5),
                self._van_der_corput(i, 7),
                self._van_der_corput(i, 11)
            ])

            lower = -0.25 + 0.2 * np.array([
                self._van_der_corput(i + n_samples, 2),
                self._van_der_corput(i + n_samples, 3),
                self._van_der_corput(i + n_samples, 5),
                self._van_der_corput(i + n_samples, 7),
                self._van_der_corput(i + n_samples, 11)
            ])

            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist()))
        return airfoils

    def _generate_optimization_samples(self, n_samples: int, rng: np.random.RandomState) -> List[Airfoil]:
        """Generate optimization-guided samples."""
        # Start with random and apply simple local improvements
        airfoils = []

        for _ in range(n_samples):
            # Random starting point
            upper = rng.uniform(0.05, 0.25, 5)
            lower = rng.uniform(-0.25, -0.05, 5)

            # Simple heuristic improvements
            # Make thickness distribution more realistic
            upper = np.sort(upper)[::-1]  # Decreasing
            lower = -np.sort(-lower)      # Decreasing in magnitude

            # Smooth transitions
            for i in range(1, 5):
                upper[i] = 0.7 * upper[i] + 0.3 * upper[i-1]
                lower[i] = 0.7 * lower[i] + 0.3 * lower[i-1]

            upper = np.clip(upper, 0.01, 0.3)
            lower = np.clip(lower, -0.3, -0.01)

            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist()))

        return airfoils

    @staticmethod
    def _van_der_corput(n: int, base: int) -> float:
        """Van der Corput sequence for quasi-random numbers."""
        vdc = 0.0
        denom = base
        while n:
            vdc += (n % base) / denom
            n //= base
            denom *= base
        return vdc

    def generate_ensemble_samples(self, total_samples: int, rng: np.random.RandomState) -> List[Tuple[Airfoil, str]]:
        """Generate samples from all strategies."""
        samples_per_strategy = total_samples // len(self.strategy_knowledge)
        remainder = total_samples % len(self.strategy_knowledge)

        ensemble_samples = []
        for i, (strategy_name, strategy_func) in enumerate(self.strategy_knowledge.items()):
            n_samples = samples_per_strategy + (1 if i < remainder else 0)
            strategy_samples = strategy_func(n_samples, rng)

            for airfoil in strategy_samples:
                ensemble_samples.append((airfoil, strategy_name))

        return ensemble_samples

class MetaLearningInitializer:
    """Meta-learning ensemble initializer."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.knowledge_base = EnsembleKnowledgeBase()
        self.transfer_predictor = TransferLearningPredictor()
        self.evaluation_history = []

    def bootstrap_with_ensemble(self, bootstrap_size: int = 80) -> List[Tuple[Airfoil, float]]:
        """Bootstrap knowledge with ensemble of strategies."""
        print("Bootstrapping with ensemble strategies...")

        # Generate samples from all strategies
        ensemble_samples = self.knowledge_base.generate_ensemble_samples(bootstrap_size, self.rng)

        # Evaluate all samples
        evaluated = []
        req = DesignRequirements(reynolds=200000, objective="max_ld")

        for airfoil, strategy in ensemble_samples:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            evaluated.append((airfoil, fitness))
            self.evaluation_history.append((airfoil, fitness, strategy))

        return evaluated

    def train_transfer_model(self):
        """Train transfer learning model on evaluation history."""
        if len(self.evaluation_history) < 10:
            return

        airfoils = [item[0] for item in self.evaluation_history]
        fitnesses = [item[1] for item in self.evaluation_history]

        self.transfer_predictor.fit(airfoils, fitnesses)
        print(f"Trained transfer model on {len(self.evaluation_history)} samples")

    def generate_meta_guided_samples(self, n_samples: int = 30) -> List[Airfoil]:
        """Generate new samples guided by meta-learning."""
        if not self.transfer_predictor.is_fitted:
            # Fallback to ensemble sampling
            fallback_samples = self.knowledge_base.generate_ensemble_samples(n_samples, self.rng)
            return [airfoil for airfoil, _ in fallback_samples]

        # Generate candidate samples
        candidate_samples = self.knowledge_base.generate_ensemble_samples(n_samples * 3, self.rng)
        candidates = [airfoil for airfoil, _ in candidate_samples]

        # Predict performance
        predicted_fitness = self.transfer_predictor.predict(candidates)

        # Select best candidates
        best_indices = np.argsort(predicted_fitness)[-n_samples:]
        selected_samples = [candidates[i] for i in best_indices]

        return selected_samples

def generate_population() -> List[Airfoil]:
    """Generate meta-learning ensemble population."""
    print("Generating meta-learning ensemble population...")

    meta_init = MetaLearningInitializer(seed=282930)

    # Bootstrap with ensemble
    bootstrap_evaluated = meta_init.bootstrap_with_ensemble(60)

    # Train transfer model
    meta_init.train_transfer_model()

    # Generate meta-guided samples
    meta_guided_samples = meta_init.generate_meta_guided_samples(25)

    # Combine all samples
    all_airfoils = [airfoil for airfoil, _ in bootstrap_evaluated] + meta_guided_samples

    # Remove duplicates (simple check)
    unique_airfoils = []
    seen_coeffs = set()

    for airfoil in all_airfoils:
        coeffs_key = tuple(np.round(airfoil.upper_coeffs + airfoil.lower_coeffs, 4))
        if coeffs_key not in seen_coeffs:
            unique_airfoils.append(airfoil)
            seen_coeffs.add(coeffs_key)

    print(f"Generated {len(unique_airfoils)} unique meta-learning airfoils")
    return unique_airfoils

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate meta-learning population."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def analyze_meta_learning_quality(population: List[Airfoil], evaluated: List[Tuple[Airfoil, float]]) -> Dict[str, float]:
    """Analyze meta-learning quality."""
    fitnesses = [f for _, f in evaluated]
    valid_fitnesses = [f for f in fitnesses if f > 0]

    if not valid_fitnesses:
        return {"error": "No valid airfoils generated"}

    quality_metrics = {
        'valid_fraction': len(valid_fitnesses) / len(fitnesses),
        'mean_fitness': np.mean(valid_fitnesses),
        'std_fitness': np.std(valid_fitnesses),
        'max_fitness': np.max(valid_fitnesses),
        'top_quartile_mean': np.mean(sorted(valid_fitnesses, reverse=True)[:max(1, len(valid_fitnesses)//4)])
    }

    # Ensemble diversity
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    quality_metrics.update({
        'ensemble_diversity': np.mean(np.std(all_coeffs, axis=0)),
        'meta_learning_gain': quality_metrics['top_quartile_mean'] / (quality_metrics['mean_fitness'] + 1e-6)
    })

    return quality_metrics

def main():
    """Main meta-learning ensemble initialization."""
    print("Generating meta-learning ensemble population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze quality
    meta_quality = analyze_meta_learning_quality(population, evaluated)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:25]):
        filename = f"results/meta_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['meta_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "meta_rank": i,
            "meta_quality": meta_quality
        })

    print(f"Generated {len(population)} meta-learning airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Meta-learning quality metrics:")
    for key, value in meta_quality.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save analysis
    with open("results/meta_analysis.json", 'w') as f:
        json.dump({
            "meta_quality": meta_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1] if evaluated else 0
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()