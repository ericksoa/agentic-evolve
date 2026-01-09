"""
Meta-learning module for predicting threshold optimization benefit.

This module provides tools to predict whether threshold optimization
will help on a given dataset, using learned meta-features instead of
hard-coded heuristics.
"""

from .extractor import MetaFeatureExtractor
from .detector import MetaLearningDetector

__all__ = ['MetaFeatureExtractor', 'MetaLearningDetector']
