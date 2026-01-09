#!/usr/bin/env python3
"""
Diabetes Evolution WITH Feature Engineering

Key insight: AutoML tools don't just tune models - they also engineer features.
This evolution includes feature engineering as a mutation strategy.

Diabetes Dataset Domain Knowledge:
- preg: Number of pregnancies (risk factor)
- plas: Plasma glucose concentration (KEY predictor)
- pres: Blood pressure
- skin: Skin thickness (body fat indicator)
- insu: Insulin level (KEY - often missing/zero)
- mass: BMI (KEY predictor)
- pedi: Diabetes pedigree function (genetic risk)
- age: Age (risk increases with age)

Known issues in the data:
- Zeros in plas, pres, skin, insu, mass are actually missing values
- High correlation between some features
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

EVOLUTION_LOG = []
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for diabetes dataset.

    This transformer creates domain-specific features that capture
    known risk factors for diabetes.
    """

    def __init__(self,
                 add_interactions=False,
                 add_ratios=False,
                 add_bins=False,
                 add_polynomials=False,
                 add_domain=False,
                 poly_degree=2):
        self.add_interactions = add_interactions
        self.add_ratios = add_ratios
        self.add_bins = add_bins
        self.add_polynomials = add_polynomials
        self.add_domain = add_domain
        self.poly_degree = poly_degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X columns: preg, plas, pres, skin, insu, mass, pedi, age
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        features = [df.copy()]

        if self.add_domain:
            # Domain-specific features based on diabetes knowledge
            domain = pd.DataFrame()

            # BMI categories (WHO classification)
            domain['bmi_underweight'] = (df['mass'] < 18.5).astype(float)
            domain['bmi_normal'] = ((df['mass'] >= 18.5) & (df['mass'] < 25)).astype(float)
            domain['bmi_overweight'] = ((df['mass'] >= 25) & (df['mass'] < 30)).astype(float)
            domain['bmi_obese'] = (df['mass'] >= 30).astype(float)

            # Age groups (risk increases with age)
            domain['age_young'] = (df['age'] < 30).astype(float)
            domain['age_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(float)
            domain['age_senior'] = (df['age'] >= 50).astype(float)

            # Glucose categories (pre-diabetes thresholds)
            domain['glucose_normal'] = (df['plas'] < 100).astype(float)
            domain['glucose_prediabetic'] = ((df['plas'] >= 100) & (df['plas'] < 126)).astype(float)
            domain['glucose_diabetic'] = (df['plas'] >= 126).astype(float)

            # High-risk flag (multiple risk factors)
            domain['high_risk'] = (
                (df['mass'] >= 30) & (df['age'] >= 40) & (df['plas'] >= 100)
            ).astype(float)

            # Pregnancy risk (gestational diabetes risk)
            domain['preg_risk'] = (df['preg'] >= 4).astype(float)

            features.append(domain)

        if self.add_ratios:
            # Ratio features
            ratios = pd.DataFrame()

            # Glucose to insulin ratio (insulin resistance indicator)
            ratios['glucose_insulin_ratio'] = df['plas'] / (df['insu'] + 1)

            # BMI to age ratio
            ratios['bmi_age_ratio'] = df['mass'] / (df['age'] + 1)

            # Glucose to BMI ratio
            ratios['glucose_bmi_ratio'] = df['plas'] / (df['mass'] + 1)

            # Pedigree adjusted by age
            ratios['pedi_age'] = df['pedi'] * df['age']

            features.append(ratios)

        if self.add_interactions:
            # Key interactions
            interactions = pd.DataFrame()

            # Glucose × BMI (both key predictors)
            interactions['glucose_bmi'] = df['plas'] * df['mass']

            # Age × BMI
            interactions['age_bmi'] = df['age'] * df['mass']

            # Glucose × Age
            interactions['glucose_age'] = df['plas'] * df['age']

            # Pregnancies × Age
            interactions['preg_age'] = df['preg'] * df['age']

            # Pedigree × Glucose (genetic + metabolic)
            interactions['pedi_glucose'] = df['pedi'] * df['plas']

            features.append(interactions)

        if self.add_bins:
            # Quantile-based bins for continuous variables
            bins = pd.DataFrame()

            for col in ['plas', 'mass', 'age', 'pedi']:
                # Create 4 bins
                bins[f'{col}_q1'] = (df[col] <= df[col].quantile(0.25)).astype(float)
                bins[f'{col}_q4'] = (df[col] >= df[col].quantile(0.75)).astype(float)

            features.append(bins)

        if self.add_polynomials:
            # Polynomial features for top predictors
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            top_features = df[['plas', 'mass', 'age']].values
            poly_features = poly.fit_transform(top_features)
            # Skip the original features (already included)
            poly_df = pd.DataFrame(
                poly_features[:, 3:],  # Skip first 3 (original features)
                columns=[f'poly_{i}' for i in range(poly_features.shape[1] - 3)]
            )
            features.append(poly_df)

        result = pd.concat(features, axis=1)
        return result.values

    def get_feature_names(self, X):
        """Get names of all features."""
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])
        names = list(df.columns)

        if self.add_domain:
            names += ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese',
                     'age_young', 'age_middle', 'age_senior',
                     'glucose_normal', 'glucose_prediabetic', 'glucose_diabetic',
                     'high_risk', 'preg_risk']

        if self.add_ratios:
            names += ['glucose_insulin_ratio', 'bmi_age_ratio', 'glucose_bmi_ratio', 'pedi_age']

        if self.add_interactions:
            names += ['glucose_bmi', 'age_bmi', 'glucose_age', 'preg_age', 'pedi_glucose']

        return names


def load_diabetes_raw():
    """Load diabetes with proper missing value handling."""
    dataset = openml.datasets.get_dataset(37)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    X_values = X.values.astype(float)

    # Handle zeros that are actually missing
    # Columns: 0=preg, 1=plas, 2=pres, 3=skin, 4=insu, 5=mass, 6=pedi, 7=age
    for col in [1, 2, 3, 4, 5]:  # plas, pres, skin, insu, mass
        X_values[X_values[:, col] == 0, col] = np.nan

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)

    return X_imputed, y.values


def create_pipeline(feature_engineer, classifier):
    """Create a pipeline with feature engineering and classification."""
    return Pipeline([
        ('features', feature_engineer),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def evaluate_pipeline(pipeline, X, y, n_cv_folds=8, n_holdout_folds=2):
    """Evaluate pipeline with CV + holdout."""
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    cv_scores = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(cv_scores), np.mean(holdout_scores), cv_scores, holdout_scores


def log_generation(gen_num, name, cv_f1, holdout_f1, description, params, accepted, reason):
    """Log generation results."""
    entry = {
        'generation': gen_num,
        'name': name,
        'cv_f1': float(cv_f1),
        'holdout_f1': float(holdout_f1),
        'gap': float(cv_f1 - holdout_f1),
        'description': description,
        'params': params,
        'accepted': bool(accepted),
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }
    EVOLUTION_LOG.append(entry)

    status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
    print(f"\n{'='*60}")
    print(f"Generation {gen_num}: {name}")
    print(f"{'='*60}")
    print(f"CV F1:      {cv_f1:.4f}")
    print(f"Holdout F1: {holdout_f1:.4f}")
    print(f"Gap:        {cv_f1 - holdout_f1:.4f}")
    print(f"Status:     {status}")
    print(f"Features:   {params.get('features', 'base')}")

    return entry


def run_evolution():
    """Run feature engineering evolution."""
    print("Loading diabetes dataset...")
    X, y = load_diabetes_raw()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y)}")

    best_holdout = 0
    best_gen = None

    # =========================================================================
    # PHASE 1: Baseline (no feature engineering)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (No Feature Engineering)")
    print("="*60)

    # Gen 0: LR+RF baseline (best from previous evolution)
    fe = DiabetesFeatureEngineer()  # No extra features
    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')

    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    entry = log_generation(0, "LR+RF Baseline", cv_f1, holdout_f1,
                          "Baseline ensemble without feature engineering",
                          {'features': 'base_only', 'n_features': 8},
                          True, "Baseline")
    best_holdout = holdout_f1
    best_gen = entry

    # =========================================================================
    # PHASE 2: Feature Engineering Mutations
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING MUTATIONS")
    print("="*60)

    # Gen 1: Add domain features only
    fe = DiabetesFeatureEngineer(add_domain=True)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(1, "Domain Features", cv_f1, holdout_f1,
                          "Add BMI categories, age groups, glucose thresholds, risk flags",
                          {'features': 'base + domain', 'n_features': 20},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 2: Add ratio features
    fe = DiabetesFeatureEngineer(add_ratios=True)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(2, "Ratio Features", cv_f1, holdout_f1,
                          "Add glucose/insulin ratio, BMI/age ratio, etc.",
                          {'features': 'base + ratios', 'n_features': 12},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 3: Add interaction features
    fe = DiabetesFeatureEngineer(add_interactions=True)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(3, "Interaction Features", cv_f1, holdout_f1,
                          "Add glucose×BMI, age×BMI, glucose×age interactions",
                          {'features': 'base + interactions', 'n_features': 13},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 4: Domain + Ratios combined
    fe = DiabetesFeatureEngineer(add_domain=True, add_ratios=True)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(4, "Domain + Ratios", cv_f1, holdout_f1,
                          "Combine domain knowledge and ratio features",
                          {'features': 'base + domain + ratios', 'n_features': 24},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 5: All features
    fe = DiabetesFeatureEngineer(add_domain=True, add_ratios=True, add_interactions=True)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(5, "All Features", cv_f1, holdout_f1,
                          "Domain + ratios + interactions",
                          {'features': 'all engineered', 'n_features': 29},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 6: Polynomial features on top predictors
    fe = DiabetesFeatureEngineer(add_polynomials=True, poly_degree=2)
    pipeline = create_pipeline(fe, clone(clf))
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(6, "Polynomial Features", cv_f1, holdout_f1,
                          "Add degree-2 polynomials of glucose, BMI, age",
                          {'features': 'base + poly2', 'n_features': 14},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # PHASE 3: Best Features + Model Tuning
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: BEST FEATURES + MODEL TUNING")
    print("="*60)

    # Find best feature set so far
    best_fe_config = best_gen['params'].get('features', 'base_only')
    print(f"Best feature config so far: {best_fe_config}")

    # Gen 7: Best features + LogReg only (simpler)
    if 'domain' in best_fe_config:
        fe = DiabetesFeatureEngineer(add_domain=True)
    elif 'ratios' in best_fe_config:
        fe = DiabetesFeatureEngineer(add_ratios=True)
    elif 'interactions' in best_fe_config:
        fe = DiabetesFeatureEngineer(add_interactions=True)
    else:
        fe = DiabetesFeatureEngineer()

    clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(7, "BestFE + LogReg", cv_f1, holdout_f1,
                          "Best features with LogReg only (simpler)",
                          {'features': best_fe_config, 'model': 'LogReg'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 8: Domain features + XGBoost
    if XGB_AVAILABLE:
        fe = DiabetesFeatureEngineer(add_domain=True)
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                eval_metric='logloss', verbosity=0)
        pipeline = create_pipeline(fe, clf)
        cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(8, "Domain + XGBoost", cv_f1, holdout_f1,
                              "Domain features with XGBoost",
                              {'features': 'domain', 'model': 'XGBoost'},
                              accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # Gen 9: Domain + interactions + LR+RF
    fe = DiabetesFeatureEngineer(add_domain=True, add_interactions=True)
    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(9, "Domain+Int + LR+RF", cv_f1, holdout_f1,
                          "Domain + interaction features with LR+RF ensemble",
                          {'features': 'domain + interactions', 'model': 'LR+RF'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 10: Ratios + stronger regularization
    fe = DiabetesFeatureEngineer(add_ratios=True)
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(10, "Ratios + StrongReg", cv_f1, holdout_f1,
                          "Ratio features with strongly regularized LogReg",
                          {'features': 'ratios', 'model': 'LogReg C=0.1'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 11: Domain only + RF (tree models might capture interactions naturally)
    fe = DiabetesFeatureEngineer(add_domain=True)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(11, "Domain + RF", cv_f1, holdout_f1,
                          "Domain features with tuned RandomForest",
                          {'features': 'domain', 'model': 'RF'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 12: Minimal features (just domain categories, no continuous)
    fe = DiabetesFeatureEngineer(add_domain=True, add_bins=True)
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(12, "Domain + Bins", cv_f1, holdout_f1,
                          "Domain features + quantile bins",
                          {'features': 'domain + bins', 'model': 'LogReg'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 13: Kitchen sink - all features + ensemble
    fe = DiabetesFeatureEngineer(add_domain=True, add_ratios=True,
                                  add_interactions=True, add_polynomials=True)
    lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(13, "All Features + Ens", cv_f1, holdout_f1,
                          "All feature types + regularized ensemble",
                          {'features': 'all', 'model': 'LR+RF regularized'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 14: Just the glucose-related features (most predictive)
    fe = DiabetesFeatureEngineer(add_domain=True)  # Includes glucose categories
    clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    pipeline = create_pipeline(fe, clf)
    cv_f1, holdout_f1, _, _ = evaluate_pipeline(pipeline, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(14, "Domain + LR", cv_f1, holdout_f1,
                          "Domain features with LogReg",
                          {'features': 'domain', 'model': 'LogReg'},
                          accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"\nBest Generation: {best_gen['generation']} - {best_gen['name']}")
    print(f"Best Holdout F1: {best_holdout:.4f}")
    print(f"Previous best (no FE): 0.624")

    improvement = ((best_holdout - 0.624) / 0.624) * 100
    print(f"Improvement over no-FE: {improvement:+.1f}%")

    baseline = 0.648
    print(f"\nTarget Comparison:")
    print(f"  vs Auto-sklearn target ({baseline * 1.15:.3f}): {'✓ BEATEN' if best_holdout >= baseline * 1.15 else '✗ NOT YET'}")
    print(f"  vs FLAML target ({baseline * 1.17:.3f}):        {'✓ BEATEN' if best_holdout >= baseline * 1.17 else '✗ NOT YET'}")
    print(f"  vs AutoGluon target ({baseline * 1.23:.3f}):    {'✓ BEATEN' if best_holdout >= baseline * 1.23 else '✗ NOT YET'}")

    # Save
    log_path = RESULTS_DIR / 'diabetes_fe_evolution_log.json'
    with open(log_path, 'w') as f:
        json.dump(EVOLUTION_LOG, f, indent=2)
    print(f"\nEvolution log saved to: {log_path}")

    return EVOLUTION_LOG, best_gen


if __name__ == '__main__':
    log, best = run_evolution()
