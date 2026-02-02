"""Progress display for evolution runs.

Provides organized, real-time progress output for parallel agent execution.
All user-visible output uses print() to stdout.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AgentProgress:
    """Track progress for a single agent."""
    variant: str
    mutation_type: str  # "mutation" or "crossover"
    parent: str
    hypothesis: str = ""
    status: AgentStatus = AgentStatus.QUEUED
    progress_pct: int = 0
    result: dict = field(default_factory=dict)
    error: str = ""


class ProgressDisplay:
    """Manages progress display for evolution runs."""

    def __init__(self, width: int = 72, use_unicode: bool = True):
        self.width = width
        self.agents: dict[str, AgentProgress] = {}

    def generation_header(self, gen: int, champion_name: str, champion_fitness: float,
                         population_size: int, plateau_count: int, plateau_threshold: int):
        """Display generation header."""
        print()
        print("=" * 60)
        print(f"  GENERATION {gen}")
        print(f"  Champion: {champion_name} ({champion_fitness:.2f}x) | Pop: {population_size} | Plateau: {plateau_count}/{plateau_threshold}")
        print("=" * 60)
        print()

    def register_agent(self, variant: str, mutation_type: str, parent: str,
                      hypothesis: str = ""):
        """Register an agent for tracking."""
        self.agents[variant] = AgentProgress(
            variant=variant,
            mutation_type=mutation_type,
            parent=parent,
            hypothesis=hypothesis,
        )

    def update_agent(self, variant: str, status: AgentStatus = None,
                    progress_pct: int = None, result: dict = None, error: str = None):
        """Update agent progress."""
        if variant not in self.agents:
            return

        agent = self.agents[variant]
        if status is not None:
            agent.status = status
        if progress_pct is not None:
            agent.progress_pct = progress_pct
        if result is not None:
            agent.result = result
        if error is not None:
            agent.error = error

    def show_agents_starting(self, agents: list[tuple[str, str, str]]):
        """Show agents that are starting.

        Args:
            agents: List of (variant, mutation_type, parent_name) tuples
        """
        print(f"  Mutations starting ({len(agents)} parallel):")
        for variant, mutation_type, parent_name in agents:
            print(f"    [{variant.upper()}] {mutation_type} (parent: {parent_name})")
        print()

    def show_agent_result(self, variant: str, mutation_type: str, fitness: float,
                         champion_fitness: float, decision: str,
                         per_size_results: dict = None, error: str = None):
        """Show result for a single agent."""
        if error:
            print(f"  [{variant.upper()}] {mutation_type[:20]:<20} FAIL  {error[:30]}")
        else:
            delta = ((fitness / champion_fitness) - 1) * 100 if champion_fitness > 0 else 0
            delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            print(f"  [{variant.upper()}] {mutation_type[:20]:<20} {fitness:.2f}x ({delta_str}) {decision}")

            # Show per-size breakdown if available
            if per_size_results and decision == "KEEP":
                for size, result in list(per_size_results.items())[:4]:
                    print(f"       {size}: {result:.2f}x")

    def generation_summary(self, gen: int, results: list[dict],
                          new_champion: dict = None, old_champion: dict = None,
                          plateau_count: int = 0):
        """Display generation summary."""
        print()
        print(f"--- Generation {gen} Summary ---")

        for r in results:
            variant = r.get("variant", "?")
            mutation_type = r.get("mutation_type", "unknown")[:18]
            fitness = r.get("fitness", 0)
            decision = r.get("decision", "?")
            error = r.get("error", "")
            trust_score = r.get("trust_score")
            original_fitness = r.get("original_fitness")

            if old_champion:
                delta = ((fitness / old_champion.get("fitness", 1)) - 1) * 100
                delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            else:
                delta_str = ""

            is_new_champ = new_champion and r.get("file") == new_champion.get("file")
            champ_marker = " <- NEW CHAMPION" if is_new_champ else ""

            # Add trust indicator if present
            trust_str = ""
            if trust_score is not None and original_fitness is not None:
                trust_str = f" [T:{trust_score:.1f}]"
                if trust_score < 1.0:
                    champ_marker = f"{champ_marker} (adj:{original_fitness:.2f}->{fitness:.2f})"

            if error:
                print(f"  [{variant.upper()}] {mutation_type:<18} FAIL   {error[:25]}")
            else:
                print(f"  [{variant.upper()}] {mutation_type:<16} {fitness:.2f}x {delta_str:>7}{trust_str} {decision}{champ_marker}")

        # Champion update
        if new_champion:
            champ_name = new_champion.get("file", "").split("/")[-1]
            champ_fitness = new_champion.get("fitness", 0)

            if old_champion and new_champion.get("file") != old_champion.get("file"):
                old_fitness = old_champion.get("fitness", 0)
                improvement = ((champ_fitness / old_fitness) - 1) * 100 if old_fitness > 0 else 0
                print(f"  Champion: {champ_name} ({champ_fitness:.2f}x) [+{improvement:.1f}% improvement]")
            else:
                print(f"  Champion: {champ_name} ({champ_fitness:.2f}x) [unchanged]")

        # Plateau status
        if plateau_count > 0:
            print(f"  Plateau: {plateau_count} generations without improvement")

        print()


BANNER = r"""
================================================================
    _                    _   _        _____            _
   / \   __ _  ___ _ __ | |_(_) ___  | ____|_   _____ | |_   _____
  / _ \ / _` |/ _ \ '_ \| __| |/ __| |  _| \ \ / / _ \| \ \ / / _ \
 / ___ \ (_| |  __/ | | | |_| | (__  | |___ \ V / (_) | |\ V /  __/
/_/   \_\__, |\___|_| |_|\__|_|\___| |_____| \_/ \___/|_| \_/ \___|
        |___/
================================================================"""


def print_evolution_header(problem: str, mode: str, work_dir: str, model: str = "",
                           max_generations: int = 0, population_size: int = 0,
                           plateau_threshold: int = 0):
    """Print the evolution start header with banner."""
    print(BANNER)
    print(f"  Problem:    {problem}")
    print(f"  Mode:       {mode}")
    if model:
        # Show short model name
        short_model = model.split("/")[-1] if "/" in model else model
        print(f"  Model:      {short_model}")
    if max_generations:
        print(f"  Generations: {max_generations} max (plateau: {plateau_threshold})")
    if population_size:
        print(f"  Population: {population_size}")
    print(f"  Work dir:   {work_dir}")
    print("=" * 64)
    print()


def print_evolution_complete(generations: int, champion: dict, population_size: int):
    """Print evolution completion summary."""
    print()
    print("=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Generations: {generations}")
    print(f"Final population: {population_size}")

    if champion:
        print(f"Champion: {champion.get('file', 'unknown')}")
        print(f"Champion fitness: {champion.get('fitness', 0):.2f}")
    else:
        print("No champion found")
    print()


def print_final_results(result: dict):
    """Print final results summary."""
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    champion = result.get("champion")
    if champion:
        print(f"\nChampion: {champion.get('file')}")
        print(f"Fitness: {champion.get('fitness')}")
        if champion.get("approach"):
            print(f"Approach: {champion.get('approach')}")
    else:
        print("\nNo champion found")

    print(f"\nGenerations run: {result.get('generations', 0)}")
    print(f"Final population size: {len(result.get('population', []))}")

    # Show fitness progression
    history = result.get("history", [])
    if history:
        print("\nFitness progression:")
        for entry in history[-10:]:  # Last 10 generations
            gen = entry.get("generation", "?")
            fitness = entry.get("best_fitness", 0)
            print(f"  Gen {gen}: {fitness:.4f}")
    print()
