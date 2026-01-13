"""Tests for the Meta-Strategist agent."""

import pytest

from evolve_sdk.agents.meta_strategist import (
    META_STRATEGIST_SYSTEM,
    get_meta_strategist_prompt,
    get_strategy_application_prompt,
    compute_mutation_effectiveness,
    compute_crossover_contribution,
    compute_diversity_index,
    should_trigger_analysis,
)


class TestMetaStrategistSystem:
    """Tests for the META_STRATEGIST_SYSTEM prompt."""

    def test_system_prompt_exists(self):
        """System prompt should be defined."""
        assert META_STRATEGIST_SYSTEM is not None
        assert len(META_STRATEGIST_SYSTEM) > 100

    def test_system_prompt_contains_key_concepts(self):
        """System prompt should cover key responsibilities."""
        assert "mutation" in META_STRATEGIST_SYSTEM.lower()
        assert "crossover" in META_STRATEGIST_SYSTEM.lower()
        assert "diversity" in META_STRATEGIST_SYSTEM.lower()
        assert "effectiveness" in META_STRATEGIST_SYSTEM.lower()

    def test_system_prompt_defines_output_format(self):
        """System prompt should specify JSON output format."""
        assert "JSON" in META_STRATEGIST_SYSTEM
        assert "analysis" in META_STRATEGIST_SYSTEM
        assert "recommendations" in META_STRATEGIST_SYSTEM

    def test_system_prompt_lists_actions(self):
        """System prompt should list possible actions."""
        assert "increase_" in META_STRATEGIST_SYSTEM
        assert "decrease_" in META_STRATEGIST_SYSTEM
        assert "inject_diversity" in META_STRATEGIST_SYSTEM


class TestGetMetaStrategistPrompt:
    """Tests for get_meta_strategist_prompt function."""

    def test_basic_prompt_generation(self):
        """Should generate a valid prompt with minimal inputs."""
        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=[],
            fitness_history=[],
        )

        assert "Generation" in prompt
        assert "10" in prompt
        assert "Mutation History" in prompt

    def test_prompt_includes_mutation_history(self):
        """Should include mutation history summary."""
        mutations = [
            {"generation": 8, "mutation_type": "structural", "success": True, "fitness_delta": 5.0},
            {"generation": 9, "mutation_type": "structural", "success": False, "fitness_delta": 0},
            {"generation": 10, "mutation_type": "parameter_tweak", "success": True, "fitness_delta": 2.0},
        ]

        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=mutations,
            crossover_history=[],
            fitness_history=[],
            analysis_window=5,
        )

        assert "structural" in prompt
        assert "parameter_tweak" in prompt

    def test_prompt_includes_crossover_history(self):
        """Should include crossover history summary."""
        crossovers = [
            {"generation": 9, "parent_fitnesses": [100, 95], "child_fitness": 105},
            {"generation": 10, "parent_fitnesses": [105, 100], "child_fitness": 102},
        ]

        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=crossovers,
            fitness_history=[],
        )

        assert "Crossover History" in prompt
        assert "Attempts" in prompt

    def test_prompt_includes_fitness_history(self):
        """Should include fitness trajectory."""
        fitness = [
            {"generation": 8, "best_fitness": 100, "avg_fitness": 80},
            {"generation": 9, "best_fitness": 110, "avg_fitness": 85},
            {"generation": 10, "best_fitness": 115, "avg_fitness": 90},
        ]

        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=[],
            fitness_history=fitness,
        )

        assert "Fitness Trajectory" in prompt
        assert "115" in prompt

    def test_prompt_includes_population_when_provided(self):
        """Should include population snapshot when provided."""
        population = [
            {"file": "gen10a.py", "fitness": 115},
            {"file": "gen10b.py", "fitness": 110},
            {"file": "gen9x.py", "fitness": 105},
        ]

        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=[],
            fitness_history=[],
            population_snapshot=population,
        )

        assert "Current Population" in prompt
        assert "gen10a.py" in prompt

    def test_prompt_respects_analysis_window(self):
        """Should filter history to analysis window."""
        mutations = [
            {"generation": 1, "mutation_type": "old", "success": True, "fitness_delta": 1},
            {"generation": 8, "mutation_type": "recent", "success": True, "fitness_delta": 5},
        ]

        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=mutations,
            crossover_history=[],
            fitness_history=[],
            analysis_window=5,
        )

        # Generation 1 is outside window (10 - 5 = 5)
        assert "recent" in prompt
        # "old" mutation type shouldn't be summarized

    def test_prompt_includes_mode_guidance(self):
        """Should include mode-specific guidance."""
        prompt_perf = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=[],
            fitness_history=[],
            mode="perf",
        )

        prompt_size = get_meta_strategist_prompt(
            generation=10,
            mutation_history=[],
            crossover_history=[],
            fitness_history=[],
            mode="size",
        )

        # Mode should be mentioned
        assert "perf" in prompt_perf
        assert "size" in prompt_size


class TestGetStrategyApplicationPrompt:
    """Tests for get_strategy_application_prompt function."""

    def test_basic_application_prompt(self):
        """Should generate prompt for applying recommendations."""
        recommendations = [
            {"action": "increase_structural_mutations", "rationale": "3x more effective", "priority": "high"},
        ]
        current_weights = {"structural": 0.3, "parameter_tweak": 0.7}

        prompt = get_strategy_application_prompt(
            recommendations=recommendations,
            current_weights=current_weights,
        )

        assert "increase_structural_mutations" in prompt
        assert "3x more effective" in prompt
        assert "structural: 0.3" in prompt

    def test_application_prompt_includes_all_recommendations(self):
        """Should include all recommendations."""
        recommendations = [
            {"action": "increase_structural_mutations", "rationale": "more effective"},
            {"action": "decrease_parameter_tweak", "rationale": "low impact"},
            {"action": "adjust_crossover_frequency", "rationale": "not contributing"},
        ]

        prompt = get_strategy_application_prompt(
            recommendations=recommendations,
            current_weights={},
        )

        assert "increase_structural_mutations" in prompt
        assert "decrease_parameter_tweak" in prompt
        assert "adjust_crossover_frequency" in prompt


class TestComputeMutationEffectiveness:
    """Tests for compute_mutation_effectiveness function."""

    def test_empty_history(self):
        """Should handle empty history."""
        result = compute_mutation_effectiveness([])
        assert result == {}

    def test_single_mutation_type(self):
        """Should calculate metrics for single type."""
        history = [
            {"mutation_type": "structural", "success": True, "fitness_delta": 10},
            {"mutation_type": "structural", "success": True, "fitness_delta": 5},
            {"mutation_type": "structural", "success": False, "fitness_delta": 0},
        ]

        result = compute_mutation_effectiveness(history)

        assert "structural" in result
        assert result["structural"]["attempts"] == 3
        assert result["structural"]["successes"] == 2
        assert result["structural"]["success_rate"] == pytest.approx(0.667, rel=0.01)
        assert result["structural"]["avg_impact"] == pytest.approx(7.5, rel=0.01)

    def test_multiple_mutation_types(self):
        """Should calculate metrics for multiple types."""
        history = [
            {"mutation_type": "structural", "success": True, "fitness_delta": 10},
            {"mutation_type": "parameter_tweak", "success": True, "fitness_delta": 2},
            {"mutation_type": "parameter_tweak", "success": False, "fitness_delta": 0},
        ]

        result = compute_mutation_effectiveness(history)

        assert "structural" in result
        assert "parameter_tweak" in result
        assert result["structural"]["effectiveness"] > result["parameter_tweak"]["effectiveness"]

    def test_negative_impact_not_counted_in_effectiveness(self):
        """Negative impact should not contribute to effectiveness."""
        history = [
            {"mutation_type": "bad_strategy", "success": True, "fitness_delta": -5},
        ]

        result = compute_mutation_effectiveness(history)

        assert result["bad_strategy"]["avg_impact"] == -5
        assert result["bad_strategy"]["effectiveness"] == 0  # Negative capped at 0

    def test_unknown_mutation_type(self):
        """Should handle missing mutation_type."""
        history = [
            {"success": True, "fitness_delta": 5},  # No mutation_type
        ]

        result = compute_mutation_effectiveness(history)
        assert "unknown" in result


class TestComputeCrossoverContribution:
    """Tests for compute_crossover_contribution function."""

    def test_empty_history(self):
        """Should handle empty history."""
        result = compute_crossover_contribution([])

        assert result["attempts"] == 0
        assert result["wins"] == 0
        assert result["contribution_rate"] == 0.0

    def test_crossover_wins(self):
        """Should count when child beats best parent."""
        history = [
            {"parent_fitnesses": [100, 95], "child_fitness": 105},  # Win
            {"parent_fitnesses": [100, 105], "child_fitness": 102},  # Loss
        ]

        result = compute_crossover_contribution(history)

        assert result["attempts"] == 2
        assert result["wins"] == 1
        assert result["contribution_rate"] == 0.5

    def test_average_improvement(self):
        """Should calculate average improvement for wins."""
        history = [
            {"parent_fitnesses": [100, 95], "child_fitness": 110},  # +10
            {"parent_fitnesses": [100, 90], "child_fitness": 120},  # +20
        ]

        result = compute_crossover_contribution(history)

        assert result["wins"] == 2
        assert result["avg_improvement"] == 15.0

    def test_no_wins(self):
        """Should handle case with no wins."""
        history = [
            {"parent_fitnesses": [100, 95], "child_fitness": 90},
            {"parent_fitnesses": [100, 95], "child_fitness": 95},
        ]

        result = compute_crossover_contribution(history)

        assert result["wins"] == 0
        assert result["contribution_rate"] == 0.0
        assert result["avg_improvement"] == 0.0


class TestComputeDiversityIndex:
    """Tests for compute_diversity_index function."""

    def test_empty_population(self):
        """Should handle empty population."""
        result = compute_diversity_index([])

        assert result["phenotypic"] == 0.0
        assert result["population_size"] == 0

    def test_single_member(self):
        """Should handle single member population."""
        result = compute_diversity_index([{"fitness": 100}])

        assert result["phenotypic"] == 0.0
        assert result["population_size"] == 1

    def test_identical_fitness(self):
        """Should return 0 diversity for identical fitness."""
        population = [
            {"fitness": 100},
            {"fitness": 100},
            {"fitness": 100},
        ]

        result = compute_diversity_index(population)

        assert result["phenotypic"] == 0.0
        assert result["fitness_range"] == 0.0

    def test_diverse_population(self):
        """Should calculate diversity for varied population."""
        population = [
            {"fitness": 100},
            {"fitness": 150},
            {"fitness": 200},
        ]

        result = compute_diversity_index(population)

        assert result["phenotypic"] > 0
        assert result["fitness_range"] == 100
        assert result["population_size"] == 3

    def test_custom_fitness_key(self):
        """Should use custom fitness key."""
        population = [
            {"score": 100},
            {"score": 200},
        ]

        result = compute_diversity_index(population, fitness_key="score")

        assert result["fitness_mean"] == 150


class TestShouldTriggerAnalysis:
    """Tests for should_trigger_analysis function."""

    def test_first_analysis(self):
        """Should trigger after reaching interval from start."""
        assert should_trigger_analysis(5, None, trigger_interval=5) is True
        assert should_trigger_analysis(4, None, trigger_interval=5) is False
        assert should_trigger_analysis(10, None, trigger_interval=5) is True

    def test_subsequent_analysis(self):
        """Should trigger based on last analysis."""
        assert should_trigger_analysis(10, 5, trigger_interval=5) is True
        assert should_trigger_analysis(9, 5, trigger_interval=5) is False
        assert should_trigger_analysis(15, 10, trigger_interval=5) is True

    def test_custom_interval(self):
        """Should respect custom trigger interval."""
        assert should_trigger_analysis(10, None, trigger_interval=10) is True
        assert should_trigger_analysis(9, None, trigger_interval=10) is False

        assert should_trigger_analysis(13, 10, trigger_interval=3) is True
        assert should_trigger_analysis(12, 10, trigger_interval=3) is False


class TestIntegration:
    """Integration tests for meta-strategist workflow."""

    def test_full_analysis_workflow(self):
        """Should handle complete analysis workflow."""
        # Simulate evolution data
        mutation_history = [
            {"generation": 6, "mutation_type": "structural", "success": True, "fitness_delta": 15},
            {"generation": 7, "mutation_type": "structural", "success": True, "fitness_delta": 10},
            {"generation": 7, "mutation_type": "parameter_tweak", "success": False, "fitness_delta": 0},
            {"generation": 8, "mutation_type": "parameter_tweak", "success": True, "fitness_delta": 2},
            {"generation": 9, "mutation_type": "structural", "success": False, "fitness_delta": 0},
            {"generation": 10, "mutation_type": "algorithm_swap", "success": True, "fitness_delta": 25},
        ]

        crossover_history = [
            {"generation": 7, "parent_fitnesses": [100, 95], "child_fitness": 105},
            {"generation": 9, "parent_fitnesses": [110, 105], "child_fitness": 108},
        ]

        fitness_history = [
            {"generation": 6, "best_fitness": 100, "avg_fitness": 80},
            {"generation": 7, "best_fitness": 110, "avg_fitness": 85},
            {"generation": 8, "best_fitness": 112, "avg_fitness": 90},
            {"generation": 9, "best_fitness": 115, "avg_fitness": 92},
            {"generation": 10, "best_fitness": 140, "avg_fitness": 100},
        ]

        population = [
            {"file": "gen10a.py", "fitness": 140},
            {"file": "gen10b.py", "fitness": 120},
            {"file": "gen9x.py", "fitness": 115},
            {"file": "gen9a.py", "fitness": 110},
        ]

        # Check trigger
        assert should_trigger_analysis(10, 5, trigger_interval=5) is True

        # Generate prompt
        prompt = get_meta_strategist_prompt(
            generation=10,
            mutation_history=mutation_history,
            crossover_history=crossover_history,
            fitness_history=fitness_history,
            population_snapshot=population,
            mode="perf",
            analysis_window=5,
        )

        # Verify prompt contains key information
        assert "structural" in prompt
        assert "algorithm_swap" in prompt
        assert "140" in prompt  # Best fitness
        assert "gen10a.py" in prompt

        # Compute metrics
        effectiveness = compute_mutation_effectiveness(mutation_history)
        assert "structural" in effectiveness
        assert "algorithm_swap" in effectiveness

        crossover_metrics = compute_crossover_contribution(crossover_history)
        assert crossover_metrics["attempts"] == 2

        diversity = compute_diversity_index(population)
        assert diversity["population_size"] == 4

    def test_low_activity_scenario(self):
        """Should handle scenarios with minimal data."""
        prompt = get_meta_strategist_prompt(
            generation=5,
            mutation_history=[
                {"generation": 5, "mutation_type": "structural", "success": True, "fitness_delta": 5},
            ],
            crossover_history=[],
            fitness_history=[
                {"generation": 5, "best_fitness": 105, "avg_fitness": 100},
            ],
            mode="size",
        )

        assert "No crossovers" in prompt
        assert "structural" in prompt
