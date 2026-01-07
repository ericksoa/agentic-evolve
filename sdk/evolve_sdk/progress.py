"""Progress display for evolution runs.

Provides organized, real-time progress output for parallel agent execution.
"""

import sys
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

    # Box-drawing characters (with ASCII fallback)
    BOX_TL = "╔"
    BOX_TR = "╗"
    BOX_BL = "╚"
    BOX_BR = "╝"
    BOX_H = "═"
    BOX_V = "║"
    BOX_TL_LIGHT = "┌"
    BOX_TR_LIGHT = "┐"
    BOX_BL_LIGHT = "└"
    BOX_BR_LIGHT = "┘"
    BOX_H_LIGHT = "─"
    BOX_V_LIGHT = "│"

    def __init__(self, width: int = 72, use_unicode: bool = True):
        self.width = width
        self.use_unicode = use_unicode
        self.agents: dict[str, AgentProgress] = {}

        if not use_unicode:
            self._use_ascii_fallback()

    def _use_ascii_fallback(self):
        """Use ASCII characters instead of Unicode."""
        self.BOX_TL = self.BOX_TR = self.BOX_BL = self.BOX_BR = "+"
        self.BOX_H = "="
        self.BOX_V = "|"
        self.BOX_TL_LIGHT = self.BOX_TR_LIGHT = "+"
        self.BOX_BL_LIGHT = self.BOX_BR_LIGHT = "+"
        self.BOX_H_LIGHT = "-"
        self.BOX_V_LIGHT = "|"

    def _box_line(self, left: str, fill: str, right: str, content: str = "") -> str:
        """Create a box line with optional content."""
        if content:
            padding = self.width - len(content) - 2
            return f"{left}{content}{' ' * padding}{right}"
        return f"{left}{fill * (self.width - 2)}{right}"

    def _progress_bar(self, pct: int, width: int = 10) -> str:
        """Create a progress bar."""
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def generation_header(self, gen: int, champion_name: str, champion_fitness: float,
                         population_size: int, plateau_count: int, plateau_threshold: int):
        """Display generation header."""
        print()
        print(self._box_line(self.BOX_TL, self.BOX_H, self.BOX_TR))

        title = f"  GENERATION {gen}"
        print(self._box_line(self.BOX_V, " ", self.BOX_V, title))

        status = f"  Champion: {champion_name} ({champion_fitness:.2f}x) | Pop: {population_size} | Plateau: {plateau_count}/{plateau_threshold}"
        print(self._box_line(self.BOX_V, " ", self.BOX_V, status))

        print(self._box_line(self.BOX_BL, self.BOX_H, self.BOX_BR))
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
        print(self._box_line(self.BOX_TL_LIGHT, self.BOX_H_LIGHT, self.BOX_TR_LIGHT))
        title = f"  MUTATIONS IN PROGRESS ({len(agents)} parallel)"
        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT, title))
        print(self._box_line(self.BOX_V_LIGHT, self.BOX_H_LIGHT, self.BOX_V_LIGHT,
                            self.BOX_H_LIGHT * (self.width - 2)))
        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT))

        # Display agents in pairs
        for i in range(0, len(agents), 2):
            left = agents[i]
            right = agents[i + 1] if i + 1 < len(agents) else None

            # First line: variant and type
            left_str = f"  [{left[0].upper()}] {left[1][:22]:<22}"
            right_str = f"  [{right[0].upper()}] {right[1][:22]:<22}" if right else ""
            print(f"{self.BOX_V_LIGHT}{left_str}{right_str:<34}{self.BOX_V_LIGHT}")

            # Second line: parent
            left_parent = f"      Parent: {left[2][:18]:<18}"
            right_parent = f"      Parent: {right[2][:18]:<18}" if right else ""
            print(f"{self.BOX_V_LIGHT}{left_parent}{right_parent:<34}{self.BOX_V_LIGHT}")

            # Third line: status (initially running)
            left_status = f"      Status: {self._progress_bar(10)} 10%"
            right_status = f"      Status: {self._progress_bar(10)} 10%" if right else ""
            print(f"{self.BOX_V_LIGHT}{left_status:<34}{right_status:<34}{self.BOX_V_LIGHT}")

            print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT))

        print(self._box_line(self.BOX_BL_LIGHT, self.BOX_H_LIGHT, self.BOX_BR_LIGHT))
        print()

    def show_agent_result(self, variant: str, mutation_type: str, fitness: float,
                         champion_fitness: float, decision: str,
                         per_size_results: dict = None, error: str = None):
        """Show result for a single agent."""
        delta = ((fitness / champion_fitness) - 1) * 100 if champion_fitness > 0 else 0
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

        status_icon = "✓" if decision == "KEEP" else "✗" if decision == "DROP" else "!"

        if error:
            print(f"  [{variant.upper()}] {mutation_type[:20]:<20} FAIL  {error[:30]}")
        else:
            print(f"  [{variant.upper()}] {mutation_type[:20]:<20} {fitness:.2f}x ({delta_str}) {decision} {status_icon}")

        # Show per-size breakdown if available
        if per_size_results and decision == "KEEP":
            for size, result in list(per_size_results.items())[:4]:
                print(f"       {size}: {result:.2f}x")

    def generation_summary(self, gen: int, results: list[dict],
                          new_champion: dict = None, old_champion: dict = None,
                          plateau_count: int = 0):
        """Display generation summary."""
        print()
        print(self._box_line(self.BOX_TL_LIGHT, self.BOX_H_LIGHT, self.BOX_TR_LIGHT))
        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT,
                            f"  GENERATION {gen} COMPLETE"))
        print(self._box_line(self.BOX_V_LIGHT, self.BOX_H_LIGHT, self.BOX_V_LIGHT,
                            self.BOX_H_LIGHT * (self.width - 2)))
        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT))

        # Results
        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT, "  Results:"))

        for r in results:
            variant = r.get("variant", "?")
            mutation_type = r.get("mutation_type", "unknown")[:18]
            fitness = r.get("fitness", 0)
            decision = r.get("decision", "?")
            error = r.get("error", "")

            if old_champion:
                delta = ((fitness / old_champion.get("fitness", 1)) - 1) * 100
                delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            else:
                delta_str = ""

            is_new_champ = new_champion and r.get("file") == new_champion.get("file")
            champ_marker = " ← NEW CHAMPION" if is_new_champ else ""

            if error:
                line = f"    [{variant.upper()}] {mutation_type:<18} FAIL   {error[:25]}"
            else:
                line = f"    [{variant.upper()}] {mutation_type:<18} {fitness:.2f}x {delta_str:>7} {decision}{champ_marker}"

            print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT, line))

        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT))

        # Champion update
        if new_champion:
            champ_name = new_champion.get("file", "").split("/")[-1]
            champ_fitness = new_champion.get("fitness", 0)

            if old_champion and new_champion.get("file") != old_champion.get("file"):
                old_fitness = old_champion.get("fitness", 0)
                improvement = ((champ_fitness / old_fitness) - 1) * 100 if old_fitness > 0 else 0
                line = f"  Champion: {champ_name} ({champ_fitness:.2f}x) [+{improvement:.1f}% improvement]"
            else:
                line = f"  Champion: {champ_name} ({champ_fitness:.2f}x) [unchanged]"

            print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT, line))

        # Plateau status
        if plateau_count > 0:
            print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT,
                                f"  Plateau: {plateau_count} generations without improvement"))
        else:
            print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT,
                                "  Plateau: RESET"))

        print(self._box_line(self.BOX_V_LIGHT, " ", self.BOX_V_LIGHT))
        print(self._box_line(self.BOX_BL_LIGHT, self.BOX_H_LIGHT, self.BOX_BR_LIGHT))
        print()

    def compact_progress(self, results: list[dict], running: list[str],
                        waiting: list[str], best_so_far: dict = None):
        """Compact progress display for many agents."""
        # Done agents
        done_str = " ".join(
            f"[{r['variant'].upper()}]{r.get('fitness', 0):.1f}x{'✓' if r.get('decision') == 'KEEP' else '✗'}"
            for r in results
        )
        print(f"  DONE: {done_str}")

        # Running agents
        run_str = " ".join(f"[{v.upper()}]████░░" for v in running)
        print(f"  RUN:  {run_str}")

        # Waiting agents
        if waiting:
            wait_str = " ".join(f"[{v.upper()}]░░░░░░" for v in waiting)
            print(f"  WAIT: {wait_str}")

        # Best so far
        if best_so_far:
            print(f"\n  Best so far: [{best_so_far.get('variant', '?').upper()}] "
                  f"{best_so_far.get('fitness', 0):.2f}x")


def print_evolution_header(problem: str, mode: str, work_dir: str):
    """Print the evolution start header."""
    print()
    print("=" * 60)
    print(f"EVOLUTION: {problem}")
    print(f"Mode: {mode}")
    print(f"Work dir: {work_dir}")
    print("=" * 60)
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
