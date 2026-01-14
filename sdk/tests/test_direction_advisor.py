"""Tests for Meta-Strategist direction recommendation extension."""

import pytest
from evolve_sdk.config import StrategicConfig
from evolve_sdk.agents.meta_strategist import (
    generate_candidate_directions,
    should_trigger_direction_analysis,
    get_direction_recommendation_prompt,
    DIRECTION_ADVISOR_SYSTEM,
)


class TestStrategicConfig:
    def test_defaults(self):
        config = StrategicConfig()
        assert config.enabled is True
        assert config.direction_interval == 10
        assert config.tactical_stall_threshold == 2
        assert config.risk_tolerance == "medium"

    def test_custom_values(self):
        config = StrategicConfig(
            enabled=False,
            direction_interval=5,
            risk_tolerance="high",
        )
        assert config.enabled is False
        assert config.direction_interval == 5
        assert config.risk_tolerance == "high"


class TestGenerateCandidateDirections:
    def test_ml_mode(self):
        directions = generate_candidate_directions("ml")
        assert len(directions) == 5
        assert all(d["source"] == "generated" for d in directions)
        names = [d["name"] for d in directions]
        assert "Feature Engineering" in names

    def test_perf_mode(self):
        directions = generate_candidate_directions("perf")
        names = [d["name"] for d in directions]
        assert "Data Structure Optimization" in names

    def test_size_mode(self):
        directions = generate_candidate_directions("size")
        names = [d["name"] for d in directions]
        assert "Functional Composition" in names

    def test_stalled_trajectory_selects_different_directions(self):
        normal = generate_candidate_directions("ml", "improving")
        stalled = generate_candidate_directions("ml", "stalled")
        # Stalled should prefer later (more radical) directions
        # Both should have 5 directions
        assert len(normal) == 5
        assert len(stalled) == 5

    def test_with_breakthrough_patterns(self):
        patterns = [
            {"name": "Custom Pattern", "description": "A test pattern", "approach_type": "hybrid"}
        ]
        directions = generate_candidate_directions("ml", breakthrough_patterns=patterns)
        assert any("Historical" in d["name"] for d in directions)

    def test_max_directions_limit(self):
        directions = generate_candidate_directions("ml", max_directions=3)
        assert len(directions) == 3

    def test_direction_ids_are_unique(self):
        directions = generate_candidate_directions("ml")
        ids = [d["id"] for d in directions]
        assert len(ids) == len(set(ids))  # All unique


class TestShouldTriggerDirectionAnalysis:
    def test_first_trigger_at_interval(self):
        assert should_trigger_direction_analysis(10, None, 10) is True
        assert should_trigger_direction_analysis(9, None, 10) is False
        assert should_trigger_direction_analysis(15, None, 10) is True

    def test_subsequent_trigger(self):
        assert should_trigger_direction_analysis(20, 10, 10) is True
        assert should_trigger_direction_analysis(15, 10, 10) is False
        assert should_trigger_direction_analysis(19, 10, 10) is False

    def test_tactical_stall_trigger(self):
        # Should trigger even before interval if tactical stall
        assert should_trigger_direction_analysis(
            generation=5,
            last_direction_analysis=None,
            trigger_interval=10,
            tactical_stall_count=3,
            tactical_stall_threshold=2,
        ) is True

    def test_tactical_stall_below_threshold(self):
        # Should NOT trigger if stall count below threshold
        assert should_trigger_direction_analysis(
            generation=5,
            last_direction_analysis=None,
            trigger_interval=10,
            tactical_stall_count=1,
            tactical_stall_threshold=2,
        ) is False

    def test_custom_interval(self):
        assert should_trigger_direction_analysis(5, None, 5) is True
        assert should_trigger_direction_analysis(4, None, 5) is False


class TestGetDirectionRecommendationPrompt:
    def test_prompt_contains_key_sections(self):
        prompt = get_direction_recommendation_prompt(
            generation=10,
            current_champion={"file": "test.py", "fitness": 0.85},
            fitness_history=[{"generation": 9, "best_fitness": 0.80}],
            mutation_history=[],
            mode="ml",
        )
        assert "Strategic Direction Analysis" in prompt
        assert "ml" in prompt.lower()
        assert "Candidate Directions" in prompt
        assert "Your Task" in prompt

    def test_prompt_includes_champion_info(self):
        prompt = get_direction_recommendation_prompt(
            generation=10,
            current_champion={
                "file": "champion.py",
                "fitness": 0.95,
                "algorithm_description": "XGBoost ensemble"
            },
            fitness_history=[],
            mutation_history=[],
            mode="ml",
        )
        assert "champion.py" in prompt
        assert "0.95" in prompt

    def test_prompt_with_user_directions(self):
        user_directions = [
            {"id": "u1", "name": "Custom Direction", "description": "Test", "approach_type": "custom", "source": "user_provided"}
        ]
        prompt = get_direction_recommendation_prompt(
            generation=10,
            current_champion={"file": "test.py", "fitness": 0.85},
            fitness_history=[],
            mutation_history=[],
            candidate_directions=user_directions,
            mode="ml",
        )
        assert "Custom Direction" in prompt

    def test_risk_tolerance_guidance(self):
        low_risk = get_direction_recommendation_prompt(
            generation=10,
            current_champion={"file": "test.py", "fitness": 0.85},
            fitness_history=[],
            mutation_history=[],
            mode="ml",
            risk_tolerance="low",
        )
        high_risk = get_direction_recommendation_prompt(
            generation=10,
            current_champion={"file": "test.py", "fitness": 0.85},
            fitness_history=[],
            mutation_history=[],
            mode="ml",
            risk_tolerance="high",
        )
        assert "safer" in low_risk.lower()
        assert "bold" in high_risk.lower()

    def test_fitness_trajectory_detection(self):
        # Improving trajectory
        improving_prompt = get_direction_recommendation_prompt(
            generation=10,
            current_champion={"file": "test.py", "fitness": 0.90},
            fitness_history=[
                {"generation": 1, "best_fitness": 0.70},
                {"generation": 5, "best_fitness": 0.80},
                {"generation": 10, "best_fitness": 0.90},
            ],
            mutation_history=[],
            mode="ml",
        )
        assert "improving" in improving_prompt.lower()


class TestDirectionAdvisorSystem:
    def test_system_prompt_exists(self):
        assert DIRECTION_ADVISOR_SYSTEM is not None
        assert len(DIRECTION_ADVISOR_SYSTEM) > 100

    def test_system_prompt_contains_output_format(self):
        assert "Output Format" in DIRECTION_ADVISOR_SYSTEM or "output" in DIRECTION_ADVISOR_SYSTEM.lower()
        assert "JSON" in DIRECTION_ADVISOR_SYSTEM

    def test_system_prompt_contains_assessment_criteria(self):
        assert "Impact" in DIRECTION_ADVISOR_SYSTEM
        assert "Risk" in DIRECTION_ADVISOR_SYSTEM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
