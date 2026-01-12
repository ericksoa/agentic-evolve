"""Trust system components for robust evolution validation.

Components:
- dossier: Trust Dossier markdown report generation
- variance: Variance Gates for statistical validation
- canary: Canary Tests for system verification
- validators: Pluggable validator interface
- exploit: Basic exploit detection (non-sandboxing)
"""

from .dossier import TrustDossier
from .variance import VarianceGate, VarianceStats
from .canary import CanaryTest, CanaryResult
from .validators import Validator, ValidatorRegistry, DefaultValidator
from .exploit import ExploitDetector, ExploitCheckResult

__all__ = [
    "TrustDossier",
    "VarianceGate",
    "VarianceStats",
    "CanaryTest",
    "CanaryResult",
    "Validator",
    "ValidatorRegistry",
    "DefaultValidator",
    "ExploitDetector",
    "ExploitCheckResult",
]
