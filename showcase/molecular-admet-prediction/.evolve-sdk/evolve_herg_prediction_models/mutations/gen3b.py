"""
Gen3-B: Random Forest with Custom Sample Weights for Imbalanced Data

Mutation from: gen0_a (Random Forest with Morgan Fingerprints)
Mutation type: Class weight adjustment
Change: Replace class_weight='balanced' with computed sample weights using
        a stronger penalty (ratio^1.5) for the minority class, which gives
        more aggressive correction for class imbalance.

Hypothesis: The default 'balanced' class weights may not be aggressive enough
for hERG data which often has significant class imbalance. By computing
explicit sample weights with a stronger penalty exponent, we can improve
recall on the minority class while maintaining reasonable precision.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


class HERGPredictor:
    """Random Forest with custom sample weights for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        # Remove class_weight='balanced' - we'll use sample_weight instead
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight=None,  # MUTATION: Using sample_weight instead
            random_state=random_state,
            n_jobs=-1
        )
        self._feature_names = None
        self._feature_importances = None
        # MUTATION: Weight exponent for minority class boost
        self._weight_exponent = 1.5

    def _calculate_features(self, smiles_list):
        """Calculate Morgan fingerprints for molecules."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(2048)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                features = np.array(fp)
            all_features.append(features)

        self._feature_names = [f'Morgan2_{i}' for i in range(2048)]
        return np.array(all_features)

    def _compute_sample_weights(self, y):
        """Compute sample weights with stronger minority class boost.

        MUTATION: Uses ratio^1.5 instead of ratio^1.0 to give stronger
        penalty for misclassifying minority class samples.
        """
        y = np.array(y)
        n_samples = len(y)
        n_classes = 2

        # Count samples per class
        class_counts = np.bincount(y, minlength=2)

        # Compute weights: (n_samples / (n_classes * count)) ^ exponent
        class_weights = {}
        for c in range(n_classes):
            if class_counts[c] > 0:
                base_weight = n_samples / (n_classes * class_counts[c])
                class_weights[c] = base_weight ** self._weight_exponent
            else:
                class_weights[c] = 1.0

        # Assign weights to each sample
        sample_weights = np.array([class_weights[yi] for yi in y])

        # Normalize so weights sum to n_samples
        sample_weights = sample_weights * n_samples / sample_weights.sum()

        return sample_weights

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings with custom sample weights."""
        X = self._calculate_features(X_smiles)
        y = np.array(y)

        # MUTATION: Compute and use custom sample weights
        sample_weights = self._compute_sample_weights(y)

        self.model.fit(X, y, sample_weight=sample_weights)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return top 30 most important features."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
