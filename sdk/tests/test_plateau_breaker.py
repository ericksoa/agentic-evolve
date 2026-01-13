"""Tests for the Plateau Breaker agent."""

import pytest
from evolve_sdk.agents.plateau_breaker import (
    PLATEAU_BREAKER_SYSTEM,
    get_plateau_breaker_prompt,
    get_intervention_prompt,
    detect_plateau,
)


class TestPlateauBreakerSystem:
    """Tests for the PLATEAU_BREAKER_SYSTEM prompt."""

    def test_system_prompt_exists(self):
        """System prompt should be a non-empty string."""
        assert isinstance(PLATEAU_BREAKER_SYSTEM, str)
        assert len(PLATEAU_BREAKER_SYSTEM) > 100

    def test_system_prompt_contains_intervention_types(self):
        """System prompt should document all intervention types."""
        intervention_types = [
            "algorithm_swap",
            "paradigm_shift",
            "data_structure",
            "decomposition",
            "external_inspiration",
        ]
        for int_type in intervention_types:
            assert int_type in PLATEAU_BREAKER_SYSTEM, f"Missing intervention type: {int_type}"

    def test_system_prompt_contains_plateau_types(self):
        """System prompt should document plateau types."""
        plateau_types = [
            "local_optimum",
            "parameter_exhaustion",
            "structural_barrier",
            "algorithmic_limit",
        ]
        for p_type in plateau_types:
            assert p_type in PLATEAU_BREAKER_SYSTEM, f"Missing plateau type: {p_type}"

    def test_system_prompt_contains_json_format(self):
        """System prompt should document expected JSON output."""
        required_fields = [
            "diagnosis",
            "interventions",
            "recommended_intervention",
            "exploration_generations",
            "message_to_mutators",
        ]
        for field in required_fields:
            assert field in PLATEAU_BREAKER_SYSTEM, f"Missing JSON field: {field}"

    def test_system_prompt_emphasizes_radical_change(self):
        """System prompt should emphasize radical over incremental."""
        assert "radical" in PLATEAU_BREAKER_SYSTEM.lower()
        assert "bold" in PLATEAU_BREAKER_SYSTEM.lower()


class TestGetPlateauBreakerPrompt:
    """Tests for the get_plateau_breaker_prompt function."""

    def test_minimal_prompt(self):
        """Should generate prompt with minimal parameters."""
        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=[],
        )

        assert isinstance(prompt, str)
        assert "/path/to/champion.py" in prompt
        assert "100.0" in prompt
        assert "Plateau Breaker" in prompt

    def test_prompt_with_fitness_history(self):
        """Should include fitness history."""
        history = [
            {"generation": 1, "best_fitness": 100.0, "avg_fitness": 90.0, "improvement_pct": 0.0},
            {"generation": 2, "best_fitness": 101.0, "avg_fitness": 92.0, "improvement_pct": 1.0},
            {"generation": 3, "best_fitness": 101.5, "avg_fitness": 93.0, "improvement_pct": 0.5},
        ]

        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 101.5},
            fitness_history=history,
            recent_mutations=[],
        )

        assert "Gen 1" in prompt
        assert "Gen 2" in prompt
        assert "Gen 3" in prompt

    def test_prompt_with_recent_mutations(self):
        """Should include recent mutation attempts."""
        mutations = [
            {"mutation_type": "parameter_tweak", "fitness_delta": 0.1, "accepted": True},
            {"mutation_type": "structural", "fitness_delta": -5.0, "accepted": False},
            {"mutation_type": "parameter_tweak", "fitness_delta": 0.05, "accepted": True},
        ]

        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=mutations,
        )

        assert "parameter_tweak" in prompt
        assert "structural" in prompt
        assert "✓" in prompt  # Accepted marker
        assert "✗" in prompt  # Rejected marker

    def test_prompt_with_failed_interventions(self):
        """Should include previously failed radical changes."""
        failed = [
            {
                "type": "algorithm_swap",
                "description": "Switched to recursive approach",
                "failure_reason": "Stack overflow on large inputs",
            }
        ]

        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=[],
            failed_interventions=failed,
        )

        assert "Previously Failed" in prompt
        assert "algorithm_swap" in prompt
        assert "Stack overflow" in prompt
        assert "AVOID" in prompt

    def test_prompt_includes_stall_info(self):
        """Should include stall detection parameters."""
        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=[],
            stall_threshold=0.02,
            stall_generations=5,
        )

        assert "2%" in prompt or "2.0%" in prompt  # Threshold percentage
        assert "5 generations" in prompt or "5" in prompt

    def test_prompt_contains_json_template(self):
        """Prompt should include JSON output template."""
        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=[],
        )

        assert '"diagnosis"' in prompt
        assert '"interventions"' in prompt
        assert '"recommended_intervention"' in prompt

    def test_prompt_with_all_parameters(self):
        """Should handle all parameters together."""
        prompt = get_plateau_breaker_prompt(
            current_champion={
                "file": "/path/to/champion.py",
                "fitness": 150.0,
                "algorithm_description": "Backtracking with pruning",
            },
            fitness_history=[
                {"generation": i, "best_fitness": 150.0 + i * 0.1, "avg_fitness": 140.0}
                for i in range(5)
            ],
            recent_mutations=[
                {"mutation_type": "tweak", "fitness_delta": 0.01, "accepted": True}
            ],
            failed_interventions=[
                {"type": "paradigm_shift", "description": "DP approach", "failure_reason": "Too slow"}
            ],
            mode="perf",
            generation=10,
            stall_threshold=0.01,
            stall_generations=4,
        )

        assert "Generation 10" in prompt
        assert "Backtracking with pruning" in prompt
        assert "150.0" in prompt


class TestGetInterventionPrompt:
    """Tests for the get_intervention_prompt function."""

    def test_intervention_prompt_basic(self):
        """Should generate prompt to execute an intervention."""
        intervention = {
            "type": "algorithm_swap",
            "description": "Replace backtracking with constraint propagation",
            "from_approach": "Backtracking search",
            "to_approach": "Arc consistency + backtracking",
            "rationale": "Prunes search space more effectively",
            "risk_level": "medium",
        }

        prompt = get_intervention_prompt(
            intervention=intervention,
            champion_file="/path/to/champion.py",
            output_file="/path/to/gen5x.py",
            mode="perf",
            generation=5,
        )

        assert "algorithm_swap" in prompt
        assert "constraint propagation" in prompt
        assert "/path/to/champion.py" in prompt
        assert "/path/to/gen5x.py" in prompt
        assert "medium" in prompt

    def test_intervention_prompt_emphasizes_radical(self):
        """Intervention prompt should emphasize radical change."""
        intervention = {
            "type": "paradigm_shift",
            "description": "Switch to iterative deepening",
            "from_approach": "BFS",
            "to_approach": "IDA*",
            "rationale": "Memory efficient",
            "risk_level": "high",
        }

        prompt = get_intervention_prompt(
            intervention=intervention,
            champion_file="/path/to/champion.py",
            output_file="/path/to/gen5x.py",
            mode="perf",
            generation=5,
        )

        assert "radical" in prompt.lower() or "RADICAL" in prompt
        assert "not a tweak" in prompt.lower() or "not an incremental" in prompt.lower()


class TestDetectPlateau:
    """Tests for the detect_plateau utility function."""

    def test_no_plateau_with_growth(self):
        """Should not detect plateau when fitness is growing."""
        history = [
            {"generation": 1, "best_fitness": 100.0},
            {"generation": 2, "best_fitness": 110.0},  # 10% growth
            {"generation": 3, "best_fitness": 121.0},  # 10% growth
            {"generation": 4, "best_fitness": 133.0},  # 10% growth
        ]

        is_stalled, gen_stalled, avg_improvement = detect_plateau(history)

        assert is_stalled is False
        assert gen_stalled == 0
        assert avg_improvement > 0.02

    def test_plateau_with_stagnation(self):
        """Should detect plateau when fitness is stagnant."""
        history = [
            {"generation": 1, "best_fitness": 100.0},
            {"generation": 2, "best_fitness": 100.5},  # 0.5% growth
            {"generation": 3, "best_fitness": 100.8},  # 0.3% growth
            {"generation": 4, "best_fitness": 101.0},  # 0.2% growth
        ]

        is_stalled, gen_stalled, avg_improvement = detect_plateau(history)

        assert is_stalled is True
        assert gen_stalled == 3
        assert avg_improvement < 0.02

    def test_plateau_with_custom_threshold(self):
        """Should respect custom threshold."""
        history = [
            {"generation": 1, "best_fitness": 100.0},
            {"generation": 2, "best_fitness": 103.0},  # 3% growth
            {"generation": 3, "best_fitness": 106.0},  # 2.9% growth
            {"generation": 4, "best_fitness": 109.0},  # 2.8% growth
        ]

        # With 2% threshold, not stalled
        is_stalled, _, _ = detect_plateau(history, threshold=0.02)
        assert is_stalled is False

        # With 5% threshold, stalled
        is_stalled, _, _ = detect_plateau(history, threshold=0.05)
        assert is_stalled is True

    def test_plateau_with_custom_window(self):
        """Should respect custom window size."""
        history = [
            {"generation": 1, "best_fitness": 100.0},
            {"generation": 2, "best_fitness": 100.1},  # Stagnant
            {"generation": 3, "best_fitness": 100.2},  # Stagnant
            {"generation": 4, "best_fitness": 100.3},  # Stagnant
            {"generation": 5, "best_fitness": 100.4},  # Stagnant
        ]

        # Window of 3
        is_stalled_3, gen_3, _ = detect_plateau(history, window=3)
        assert is_stalled_3 is True
        assert gen_3 == 3

        # Window of 4
        is_stalled_4, gen_4, _ = detect_plateau(history, window=4)
        assert is_stalled_4 is True
        assert gen_4 == 4

    def test_insufficient_history(self):
        """Should handle insufficient history gracefully."""
        # Only 2 generations, need at least window+1
        history = [
            {"generation": 1, "best_fitness": 100.0},
            {"generation": 2, "best_fitness": 100.1},
        ]

        is_stalled, gen_stalled, avg = detect_plateau(history, window=3)

        assert is_stalled is False
        assert gen_stalled == 0

    def test_empty_history(self):
        """Should handle empty history gracefully."""
        is_stalled, gen_stalled, avg = detect_plateau([])

        assert is_stalled is False
        assert gen_stalled == 0
        assert avg == 0.0

    def test_zero_fitness_handling(self):
        """Should handle zero fitness values without division errors."""
        history = [
            {"generation": 1, "best_fitness": 0.0},
            {"generation": 2, "best_fitness": 0.0},
            {"generation": 3, "best_fitness": 0.0},
            {"generation": 4, "best_fitness": 1.0},
        ]

        # Should not raise division by zero
        is_stalled, _, _ = detect_plateau(history)
        # Result doesn't matter, just shouldn't crash


class TestPlateauBreakerIntegration:
    """Integration tests for plateau breaker with memory and runner."""

    def test_output_format_compatible_with_memory(self):
        """Plateau breaker output should be storable in memory."""
        # The interventions should be structured for memory storage
        assert "interventions" in PLATEAU_BREAKER_SYSTEM
        assert "type" in PLATEAU_BREAKER_SYSTEM
        assert "description" in PLATEAU_BREAKER_SYSTEM

    def test_message_to_mutators_included(self):
        """Should include guidance for next generation mutators."""
        prompt = get_plateau_breaker_prompt(
            current_champion={"file": "/path/to/champion.py", "fitness": 100.0},
            fitness_history=[],
            recent_mutations=[],
        )

        assert "message_to_mutators" in prompt

    def test_intervention_can_be_passed_to_mutator(self):
        """Intervention prompt should be usable by mutator agent."""
        intervention = {
            "type": "algorithm_swap",
            "description": "Test intervention",
            "from_approach": "A",
            "to_approach": "B",
            "rationale": "Testing",
            "risk_level": "low",
        }

        prompt = get_intervention_prompt(
            intervention=intervention,
            champion_file="/path/to/champion.py",
            output_file="/path/to/output.py",
            mode="perf",
            generation=1,
        )

        # Should include file paths for agent to read/write
        assert "champion" in prompt.lower()
        assert "output" in prompt.lower()
        assert "Return JSON" in prompt
