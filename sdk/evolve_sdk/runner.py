"""
Main evolution runner using Claude Agent SDK.

Provides hierarchical agents with clean context per generation,
parallel mutation exploration, and validation hooks.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# NOTE: claude_agent_sdk import is conditional - allows testing structure without SDK installed
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Stub for development/testing without SDK
    async def query(*args, **kwargs):
        raise RuntimeError("Claude Agent SDK not installed. Run: pip install claude-agent-sdk")
    ClaudeAgentOptions = dict
    HookMatcher = dict

from .config import EvolutionConfig
from .agents import (
    INITIALIZER_SYSTEM, get_initializer_prompt,
    MUTATOR_SYSTEM, get_mutator_prompt,
    CROSSOVER_SYSTEM, get_crossover_prompt,
    EVALUATOR_SYSTEM, get_evaluator_prompt,
)
from .hooks import create_validation_hook, create_logging_hook, set_log_context
from .progress import (
    ProgressDisplay, AgentStatus,
    print_evolution_header, print_evolution_complete, print_final_results,
)


class EvolutionRunner:
    """
    Orchestrates evolution using hierarchical Claude agents.

    Each generation spawns dedicated subagents with clean context:
    - Initializer: Creates diverse starting population
    - Mutators: Create variants of top solutions (parallel)
    - Crossover: Combines best solutions
    - Evaluator: Measures fitness of all solutions

    State is persisted to disk, allowing resume across sessions.
    """

    def __init__(self, problem: str, mode: str = "size", **kwargs):
        """
        Initialize the evolution runner.

        Args:
            problem: Description of what to evolve
            mode: Optimization mode (size, perf, ml)
            **kwargs: Additional config options (see EvolutionConfig)
                - test_command: Custom evaluation command (e.g., "python evaluate.py {solution}")
                - starter_solutions: List of paths to seed solutions
                - optimization_strategies: List of optimization hints
                - cwd: Working directory for subagents
                - env: Environment variables for subagents
        """
        # Extract extra kwargs not in EvolutionConfig
        self.test_command = kwargs.pop("test_command", None)
        self.starter_solutions = kwargs.pop("starter_solutions", [])
        self.optimization_strategies = kwargs.pop("optimization_strategies", [])
        self.cwd = kwargs.pop("cwd", None)
        self.env = kwargs.pop("env", {})

        self.config = EvolutionConfig(problem=problem, mode=mode, **kwargs)
        # Also store test_command in config for evaluator access
        if self.test_command:
            self.config.test_command = self.test_command
        self.problem_id = self._sanitize_problem_id(problem)
        self.work_dir = self.config.evolve_dir / self.problem_id
        self.mutations_dir = self.work_dir / "mutations"

        self.generation = 0
        self.population: list[dict[str, Any]] = []
        self.champion: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

        # Logging
        self.log_file = self.work_dir / "tool_usage.jsonl"

        # Progress display
        self.progress = ProgressDisplay(width=72)

    def _sanitize_problem_id(self, problem: str) -> str:
        """Convert problem description to valid directory name."""
        # Take first 30 chars, replace non-alphanumeric with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", problem[:30].lower())
        return re.sub(r"_+", "_", sanitized).strip("_")

    async def run(self) -> dict[str, Any]:
        """
        Run the full evolution loop.

        Returns:
            Final results including champion solution
        """
        print_evolution_header(self.config.problem, self.config.mode, str(self.work_dir))

        # Setup directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.mutations_dir.mkdir(parents=True, exist_ok=True)

        # Check for resume
        if self._load_state():
            print(f"Resumed from generation {self.generation}")
        else:
            # Phase 1: Initialize population
            await self._initialize_population()

        # Phase 2: Evolution loop
        plateau_count = 0
        best_fitness = self.champion["fitness"] if self.champion else 0

        while (
            self.generation < self.config.max_generations
            and plateau_count < self.config.plateau_threshold
        ):
            self.generation += 1
            set_log_context(self.log_file, self.generation)

            # Display generation header
            champion_name = self.champion.get("file", "").split("/")[-1] if self.champion else "none"
            champion_fitness = self.champion.get("fitness", 0) if self.champion else 0
            self.progress.generation_header(
                gen=self.generation,
                champion_name=champion_name,
                champion_fitness=champion_fitness,
                population_size=len(self.population),
                plateau_count=plateau_count,
                plateau_threshold=self.config.plateau_threshold,
            )

            # Run one generation
            gen_best = await self._run_generation()

            # Track improvement
            if gen_best and gen_best["fitness"] > best_fitness:
                improvement = gen_best["fitness"] - best_fitness
                print(f"[+] Improvement: {best_fitness:.4f} -> {gen_best['fitness']:.4f} (+{improvement:.4f})")
                best_fitness = gen_best["fitness"]
                plateau_count = 0
            else:
                plateau_count += 1
                print(f"[=] No improvement (plateau: {plateau_count}/{self.config.plateau_threshold})")

            self._save_state()

        # Phase 3: Finalize
        return await self._finalize()

    async def _initialize_population(self):
        """Bootstrap initial population with dedicated agent."""
        print("\n[Gen 0] Initializing population...")

        # Use absolute path for output_dir so subagents can find it
        output_dir = self.mutations_dir.resolve() if self.mutations_dir.is_absolute() else (Path(self.cwd or ".") / self.mutations_dir).resolve()

        prompt = get_initializer_prompt(
            problem=self.config.problem,
            mode=self.config.mode,
            output_dir=str(output_dir),
            population_size=self.config.population_size,
        )

        result = await self._run_agent(
            system=INITIALIZER_SYSTEM,
            prompt=prompt,
            tools=["Read", "Write", "Bash", "Glob"],
            max_turns=25,
        )

        # Parse results
        parsed = self._parse_json_from_result(result)
        if parsed and "solutions" in parsed:
            self.population = parsed["solutions"]
            if parsed.get("best"):
                self.champion = parsed["best"]
            print(f"[Gen 0] Created {len(self.population)} initial solutions")
            print(f"[Gen 0] Best initial fitness: {self.champion['fitness'] if self.champion else 'N/A'}")
        else:
            print("[Gen 0] Warning: Could not parse initialization results")
            # Try to discover any created files
            await self._discover_population()

        self.generation = 0
        self._save_state()

    async def _run_generation(self) -> dict[str, Any] | None:
        """
        Run one generation of evolution.

        1. Select top performers
        2. Create mutations (parallel)
        3. Create crossover
        4. Evaluate all new solutions
        5. Update population

        Returns:
            Best solution from this generation
        """
        # Sort population by fitness
        self.population.sort(key=lambda x: x.get("fitness", 0), reverse=True)
        top_n = self.population[: self.config.elite_count]

        if not top_n:
            print("[!] No population to evolve from")
            return None

        # Build agent list for display
        agents_info = []
        mutation_tasks = []
        task_variants = []

        for i, parent in enumerate(top_n):
            if i >= self.config.mutation_variants:
                break
            variant = chr(ord("a") + i)
            output_file = str(self.mutations_dir / f"gen{self.generation}{variant}.py")
            parent_name = parent.get("file", "").split("/")[-1]
            agents_info.append((variant, "mutation", parent_name))
            task = self._spawn_mutator(parent, output_file, variant)
            mutation_tasks.append(task)
            task_variants.append(variant)

        # Also spawn crossover
        crossover_file = str(self.mutations_dir / f"gen{self.generation}x.py")
        crossover_task = self._spawn_crossover(top_n, crossover_file)
        mutation_tasks.append(crossover_task)
        task_variants.append("x")
        parent_names = "+".join(p.get("file", "").split("/")[-1][:8] for p in top_n[:2])
        agents_info.append(("x", "crossover", parent_names))

        # Show agents starting
        self.progress.show_agents_starting(agents_info)

        # Run mutations in parallel (or sequential if configured)
        if self.config.parallel_mutations:
            mutations = await asyncio.gather(*mutation_tasks, return_exceptions=True)
        else:
            mutations = []
            for task in mutation_tasks:
                try:
                    result = await task
                    mutations.append(result)
                except Exception as e:
                    mutations.append(e)

        # Filter out exceptions and track results
        valid_mutations = []
        gen_results = []

        for i, (m, variant) in enumerate(zip(mutations, task_variants)):
            if isinstance(m, Exception):
                gen_results.append({
                    "variant": variant,
                    "mutation_type": agents_info[i][1],
                    "decision": "DROP",
                    "error": str(m)[:50],
                })
            elif isinstance(m, dict) and m.get("file"):
                valid_mutations.append(m)
                m["variant"] = variant
                m["mutation_type"] = agents_info[i][1]
            else:
                gen_results.append({
                    "variant": variant,
                    "mutation_type": agents_info[i][1],
                    "decision": "DROP",
                    "error": "no file created",
                })

        # Evaluate all new solutions
        old_champion = self.champion.copy() if self.champion else None

        if valid_mutations:
            files_to_eval = [m["file"] for m in valid_mutations]
            evaluations = await self._spawn_evaluator(files_to_eval)

            # Match evaluations back to mutations and build results
            eval_by_file = {e.get("file"): e for e in evaluations}
            for m in valid_mutations:
                eval_result = eval_by_file.get(m["file"], {})
                fitness = eval_result.get("fitness", 0)
                is_valid = eval_result.get("valid", False)

                # Determine decision
                champion_fitness = old_champion.get("fitness", 0) if old_champion else 0
                if is_valid and fitness > 0:
                    decision = "KEEP"
                    # Add to population
                    combined = {**m, **eval_result}
                    self.population.append(combined)
                else:
                    decision = "DROP"

                gen_results.append({
                    "variant": m.get("variant", "?"),
                    "mutation_type": m.get("mutation_type", "unknown"),
                    "fitness": fitness,
                    "decision": decision,
                    "file": m.get("file"),
                    "error": eval_result.get("error", "") if not is_valid else "",
                })

        # Selection: keep top N
        self.population.sort(key=lambda x: x.get("fitness", 0), reverse=True)
        self.population = self.population[: self.config.population_size]

        # Update champion
        new_champion = None
        if self.population:
            best = self.population[0]
            if not self.champion or best["fitness"] > self.champion["fitness"]:
                self.champion = best.copy()
                new_champion = self.champion
                self._save_champion()

        # Show generation summary
        self.progress.generation_summary(
            gen=self.generation,
            results=gen_results,
            new_champion=new_champion,
            old_champion=old_champion,
            plateau_count=0,  # Will be updated by caller
        )

        # Record history
        kept_count = sum(1 for r in gen_results if r.get("decision") == "KEEP")
        self.history.append({
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(self.population),
            "best_fitness": self.champion["fitness"] if self.champion else 0,
            "mutations_tried": len(mutation_tasks),
            "mutations_valid": kept_count,
        })

        return self.champion

    async def _spawn_mutator(
        self, parent: dict, output_file: str, variant: str
    ) -> dict[str, Any]:
        """Spawn a dedicated mutator agent (clean context)."""
        prompt = get_mutator_prompt(
            parent_file=parent["file"],
            parent_fitness=parent.get("fitness", 0),
            output_file=output_file,
            mode=self.config.mode,
            generation=self.generation,
            variant=variant,
        )

        result = await self._run_agent(
            system=MUTATOR_SYSTEM,
            prompt=prompt,
            tools=["Read", "Write", "Edit", "Bash"],
            max_turns=self.config.max_turns_per_agent,
        )

        parsed = self._parse_json_from_result(result)
        return parsed if parsed else {"file": None, "error": "Failed to parse"}

    async def _spawn_crossover(
        self, parents: list[dict], output_file: str
    ) -> dict[str, Any]:
        """Spawn a dedicated crossover agent (clean context)."""
        prompt = get_crossover_prompt(
            parent_files=[p["file"] for p in parents],
            parent_fitnesses=[p.get("fitness", 0) for p in parents],
            output_file=output_file,
            mode=self.config.mode,
            generation=self.generation,
        )

        result = await self._run_agent(
            system=CROSSOVER_SYSTEM,
            prompt=prompt,
            tools=["Read", "Write", "Edit"],
            max_turns=self.config.max_turns_per_agent,
        )

        parsed = self._parse_json_from_result(result)
        return parsed if parsed else {"file": None, "error": "Failed to parse"}

    async def _spawn_evaluator(self, files: list[str]) -> list[dict[str, Any]]:
        """Spawn a dedicated evaluator agent (clean context)."""
        prompt = get_evaluator_prompt(
            solution_files=files,
            mode=self.config.mode,
            benchmark_command=self.config.test_command,
        )

        result = await self._run_agent(
            system=EVALUATOR_SYSTEM,
            prompt=prompt,
            tools=["Read", "Bash", "Glob"],
            max_turns=self.config.max_turns_per_agent + 5,  # Extra turns for multiple evals
        )

        parsed = self._parse_json_from_result(result)
        if parsed and "evaluations" in parsed:
            return parsed["evaluations"]
        return []

    async def _run_agent(
        self,
        system: str,
        prompt: str,
        tools: list[str],
        max_turns: int = 15,
    ) -> str:
        """
        Run a single agent query with clean context.

        Each call creates a fresh agent - no session resume.
        This ensures each subagent has clean, focused context.
        """
        if not SDK_AVAILABLE:
            raise RuntimeError(
                "Claude Agent SDK not installed. Run: pip install claude-agent-sdk"
            )

        result_text = ""

        # Note: hooks disabled for now due to SDK compatibility issues
        # TODO: Re-enable when claude_agent_sdk hook interface is clarified

        try:
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    system_prompt=system,
                    allowed_tools=tools,
                    permission_mode="acceptEdits",
                    max_turns=max_turns,
                    model=self.config.model,
                    cwd=self.cwd,
                ),
            ):
                # Collect result text
                if hasattr(message, "content"):
                    result_text += str(message.content)
                elif hasattr(message, "result"):
                    result_text += str(message.result)
                elif isinstance(message, str):
                    result_text += message

        except Exception as e:
            print(f"[!] Agent error: {e}")
            return f'{{"error": "{str(e)}"}}'

        return result_text

    async def _finalize(self) -> dict[str, Any]:
        """Finalize evolution and return results."""
        print_evolution_complete(self.generation, self.champion, len(self.population))
        self._save_state()

        return {
            "problem": self.config.problem,
            "mode": self.config.mode,
            "generations": self.generation,
            "champion": self.champion,
            "population": self.population,
            "history": self.history,
        }

    async def _discover_population(self):
        """Discover existing solution files if parsing failed."""
        import glob

        # Use absolute path for discovery
        mutations_path = self.mutations_dir.resolve() if self.mutations_dir.is_absolute() else (Path(self.cwd or ".") / self.mutations_dir).resolve()
        pattern = str(mutations_path / "gen0_*.py")
        files = glob.glob(pattern)

        print(f"[Gen 0] Discovering files in: {mutations_path}")
        print(f"[Gen 0] Found {len(files)} files")

        for f in files:
            self.population.append({
                "file": f,
                "fitness": 0,
                "approach": "discovered",
            })

    def _parse_json_from_result(self, text: str) -> dict | list | None:
        """Extract JSON from agent response text."""
        if not text:
            return None

        # Try to find JSON in the text
        # Look for {...} or [...]
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # Markdown code block
            r"```\s*([\s\S]*?)\s*```",  # Generic code block
            r"(\{[\s\S]*\})",  # Raw JSON object
            r"(\[[\s\S]*\])",  # Raw JSON array
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _load_state(self) -> bool:
        """Load state from disk. Returns True if state was loaded."""
        state_file = self.work_dir / "evolution.json"
        if not state_file.exists():
            return False

        try:
            state = json.loads(state_file.read_text())
            self.generation = state.get("generation", 0)
            self.population = state.get("population", [])
            self.champion = state.get("champion")
            self.history = state.get("history", [])
            return True
        except Exception as e:
            print(f"[!] Could not load state: {e}")
            return False

    def _save_state(self):
        """Save current state to disk."""
        state = {
            "problem": self.config.problem,
            "mode": self.config.mode,
            "generation": self.generation,
            "population": self.population,
            "champion": self.champion,
            "history": self.history,
            "config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size,
                "elite_count": self.config.elite_count,
            },
            "updated_at": datetime.now().isoformat(),
        }

        state_file = self.work_dir / "evolution.json"
        state_file.write_text(json.dumps(state, indent=2))

    def _save_champion(self):
        """Save champion to separate file for easy access."""
        if not self.champion:
            return

        champion_file = self.work_dir / "champion.json"
        champion_file.write_text(json.dumps(self.champion, indent=2))
