"""Tests for simplified progress display functions."""

import pytest
from evolve_sdk.progress import (
    ProgressDisplay,
    AgentStatus,
    AgentProgress,
    print_evolution_header,
    print_evolution_complete,
    print_final_results,
    BANNER,
)


class TestProgressDisplay:
    """Test the simplified ProgressDisplay class."""

    def test_generation_header(self, capsys):
        """generation_header should print clean text without box drawing."""
        pd = ProgressDisplay()
        pd.generation_header(
            gen=3, champion_name="gen2a.py", champion_fitness=18.72,
            population_size=10, plateau_count=1, plateau_threshold=5,
        )
        captured = capsys.readouterr()
        assert "GENERATION 3" in captured.out
        assert "gen2a.py" in captured.out
        assert "18.72" in captured.out
        assert "1/5" in captured.out
        # No box-drawing characters
        for ch in ["╔", "╗", "╚", "╝", "═", "║", "┌", "┐", "└", "┘", "─", "│"]:
            assert ch not in captured.out

    def test_show_agents_starting(self, capsys):
        """show_agents_starting should print clean agent list."""
        pd = ProgressDisplay()
        agents = [
            ("a", "mutation", "gen1a.py"),
            ("b", "mutation", "gen1b.py"),
            ("x", "crossover", "gen1a+gen1b"),
        ]
        pd.show_agents_starting(agents)
        captured = capsys.readouterr()
        assert "2 mutations + 1 crossover" in captured.out
        assert "[A]" in captured.out
        assert "gen1a.py" in captured.out
        assert "combining" in captured.out
        # No box-drawing
        assert "║" not in captured.out
        assert "█" not in captured.out

    def test_show_agents_starting_with_fitness(self, capsys):
        """show_agents_starting should display fitness when provided."""
        pd = ProgressDisplay()
        agents = [
            ("a", "mutation", "gen1a.py", 1.50),
            ("x", "crossover", "gen1a+gen1b", 1.50),
        ]
        pd.show_agents_starting(agents)
        captured = capsys.readouterr()
        assert "1.50x" in captured.out

    def test_show_agent_result_success(self, capsys):
        """show_agent_result should print fitness and decision."""
        pd = ProgressDisplay()
        pd.show_agent_result(
            variant="a", mutation_type="parameter_tweak",
            fitness=20.0, champion_fitness=18.72,
            decision="KEEP",
        )
        captured = capsys.readouterr()
        assert "[A]" in captured.out
        assert "20.00x" in captured.out
        assert "KEEP" in captured.out

    def test_show_agent_result_failure(self, capsys):
        """show_agent_result with error should print FAIL."""
        pd = ProgressDisplay()
        pd.show_agent_result(
            variant="b", mutation_type="algorithm_swap",
            fitness=0, champion_fitness=18.72,
            decision="DROP", error="syntax error",
        )
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "syntax error" in captured.out

    def test_generation_summary(self, capsys):
        """generation_summary should print formatted results."""
        pd = ProgressDisplay()
        results = [
            {"variant": "a", "mutation_type": "param_tweak", "fitness": 20.0, "decision": "KEEP"},
            {"variant": "b", "mutation_type": "algo_swap", "fitness": 12.0, "decision": "DROP"},
        ]
        old_champion = {"fitness": 18.72, "file": "/old.py"}
        new_champion = {"fitness": 20.0, "file": "/new.py"}
        pd.generation_summary(
            gen=3, results=results,
            new_champion=new_champion, old_champion=old_champion,
        )
        captured = capsys.readouterr()
        assert "Generation 3 Summary" in captured.out
        assert "[A]" in captured.out
        assert "[B]" in captured.out
        # No box-drawing
        for ch in ["╔", "╗", "╚", "╝", "═", "║"]:
            assert ch not in captured.out

    def test_generation_summary_with_trust(self, capsys):
        """generation_summary should show trust info when present."""
        pd = ProgressDisplay()
        results = [
            {
                "variant": "a", "mutation_type": "mutation",
                "fitness": 15.0, "decision": "KEEP",
                "trust_score": 0.8, "original_fitness": 18.0,
            },
        ]
        old_champion = {"fitness": 14.0, "file": "/old.py"}
        pd.generation_summary(gen=2, results=results, old_champion=old_champion)
        captured = capsys.readouterr()
        assert "T:0.8" in captured.out


class TestBanner:
    """Test the ASCII art banner."""

    def test_banner_contains_agentic_evolve(self):
        """Banner should contain recognizable text art."""
        assert "Agentic" in BANNER or "___ _ __ | |_(_)" in BANNER

    def test_banner_no_ansi(self):
        """Banner should not contain ANSI escape codes."""
        assert "\033[" not in BANNER


class TestFreeStandingFunctions:
    """Test the backward-compatible free functions."""

    def test_print_evolution_header(self, capsys):
        """print_evolution_header should show banner and problem info."""
        print_evolution_header("shortest sort", "size", "/tmp/work")
        captured = capsys.readouterr()
        assert "shortest sort" in captured.out
        assert "size" in captured.out
        assert "/tmp/work" in captured.out
        # Should include banner
        assert "___" in captured.out

    def test_print_evolution_header_with_model(self, capsys):
        """print_evolution_header should show model when provided."""
        print_evolution_header("test", "perf", "/tmp", model="claude-opus")
        captured = capsys.readouterr()
        assert "claude-opus" in captured.out

    def test_print_evolution_complete(self, capsys):
        """print_evolution_complete should show summary."""
        champion = {"file": "/best.py", "fitness": 42.5}
        print_evolution_complete(10, champion, 8)
        captured = capsys.readouterr()
        assert "EVOLUTION COMPLETE" in captured.out
        assert "10" in captured.out
        assert "42.50" in captured.out

    def test_print_evolution_complete_no_champion(self, capsys):
        """print_evolution_complete should handle no champion."""
        print_evolution_complete(5, None, 0)
        captured = capsys.readouterr()
        assert "No champion found" in captured.out

    def test_print_final_results(self, capsys):
        """print_final_results should show results summary."""
        result = {
            "champion": {"file": "/champ.py", "fitness": 99.9},
            "generations": 25,
            "population": [{}] * 5,
            "history": [
                {"generation": 1, "best_fitness": 10.0},
                {"generation": 2, "best_fitness": 20.0},
            ],
        }
        print_final_results(result)
        captured = capsys.readouterr()
        assert "FINAL RESULTS" in captured.out
        assert "/champ.py" in captured.out
        assert "99.9" in captured.out
        assert "Gen 1" in captured.out

    def test_no_ansi_in_any_output(self, capsys):
        """No function should produce ANSI escape codes."""
        pd = ProgressDisplay()
        pd.generation_header(1, "test.py", 1.0, 5, 0, 5)
        pd.show_agents_starting([("a", "mutation", "p.py")])
        pd.show_agent_result("a", "mutation", 2.0, 1.0, "KEEP")
        pd.generation_summary(1, [{"variant": "a", "mutation_type": "m", "fitness": 2.0, "decision": "KEEP"}])
        print_evolution_header("test", "size", "/tmp")
        print_evolution_complete(1, {"file": "x", "fitness": 1.0}, 1)
        print_final_results({"champion": None, "generations": 0, "population": [], "history": []})

        captured = capsys.readouterr()
        assert "\033[" not in captured.out


class TestDataStructures:
    """Test that data structures are preserved."""

    def test_agent_status_enum(self):
        assert AgentStatus.QUEUED.value == "queued"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.DONE.value == "done"
        assert AgentStatus.FAILED.value == "failed"

    def test_agent_progress_dataclass(self):
        ap = AgentProgress(variant="a", mutation_type="mutation", parent="gen1.py")
        assert ap.variant == "a"
        assert ap.status == AgentStatus.QUEUED
        assert ap.error == ""
