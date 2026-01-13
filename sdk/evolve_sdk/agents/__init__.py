"""Agent prompt definitions for evolution subagents."""

from .mutator import MUTATOR_SYSTEM, get_mutator_prompt
from .evaluator import EVALUATOR_SYSTEM, get_evaluator_prompt
from .crossover import CROSSOVER_SYSTEM, get_crossover_prompt
from .initializer import INITIALIZER_SYSTEM, get_initializer_prompt
from .adversary import ADVERSARY_SYSTEM, get_adversary_prompt, get_escalation_prompt
from .arbitrator import ARBITRATOR_SYSTEM, get_arbitrator_prompt
from .reporter import ReporterAgent, report_to_operator
from .debugger import DEBUGGER_SYSTEM, get_debugger_prompt, get_debugger_summary_prompt
from .plateau_breaker import (
    PLATEAU_BREAKER_SYSTEM,
    get_plateau_breaker_prompt,
    get_intervention_prompt,
    detect_plateau,
)

__all__ = [
    "MUTATOR_SYSTEM", "get_mutator_prompt",
    "EVALUATOR_SYSTEM", "get_evaluator_prompt",
    "CROSSOVER_SYSTEM", "get_crossover_prompt",
    "INITIALIZER_SYSTEM", "get_initializer_prompt",
    "ADVERSARY_SYSTEM", "get_adversary_prompt", "get_escalation_prompt",
    "ARBITRATOR_SYSTEM", "get_arbitrator_prompt",
    "ReporterAgent", "report_to_operator",
    "DEBUGGER_SYSTEM", "get_debugger_prompt", "get_debugger_summary_prompt",
    "PLATEAU_BREAKER_SYSTEM", "get_plateau_breaker_prompt", "get_intervention_prompt", "detect_plateau",
]
