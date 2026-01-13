#!/usr/bin/env python3
"""Test script for gen1x crossover hybrid."""

from mutations.gen1x import HERGPredictor
import numpy as np

# Test instantiation
predictor = HERGPredictor(random_state=42)
print('Model instantiation: OK')

# Test with sample SMILES (common drug molecules)
test_smiles = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
]
test_labels = [0, 1, 0, 1]

# Test feature calculation
features = predictor._calculate_features(test_smiles)
print(f'Feature shape: {features.shape}')
print(f'Expected: (4, 714)')

# Test fit
predictor.fit(test_smiles, test_labels)
print('Model fitting: OK')

# Test predict_proba
proba = predictor.predict_proba(test_smiles)
print(f'Predictions shape: {proba.shape}')
print(f'Sample predictions: {proba}')

# Test predict
preds = predictor.predict(test_smiles)
print(f'Binary predictions: {preds}')

# Test feature importance
importance = predictor.get_feature_importance()
print(f'Top 5 features: {importance["features"][:5]}')

print('\nAll tests passed!')
