"""
Main evolution runner using Claude Agent SDK.

Provides hierarchical agents with clean context per generation,
parallel mutation exploration, and validation hooks.

Trust System Components:
- Variance Gates: Re-evaluate N times for consistency
- Canary Tests: Inject known-bad candidates to verify system works
- Exploit Detection: Timing, output integrity, determinism checks
- Trust Dossier: Markdown reports of trust decisions
- Validators: Pluggable validation chain for escalation
"""

import asyncio
import json
import re
import time
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

from .config import EvolutionConfig, TrustConfig
from .agents import (
    INITIALIZER_SYSTEM, get_initializer_prompt,
    MUTATOR_SYSTEM, get_mutator_prompt,
    CROSSOVER_SYSTEM, get_crossover_prompt,
    EVALUATOR_SYSTEM, get_evaluator_prompt,
    ADVERSARY_SYSTEM, get_adversary_prompt, get_escalation_prompt,
    ARBITRATOR_SYSTEM, get_arbitrator_prompt,
)
from .hooks import create_validation_hook, create_logging_hook, set_log_context
from .progress import (
    ProgressDisplay, AgentStatus,
    print_evolution_header, print_evolution_complete, print_final_results,
)
from .trust import (
    TrustDossier,
    VarianceGate,
    CanaryTest,
    ExploitDetector,
    ValidatorRegistry,
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

        # Trust system components
        self._init_trust_system()

    def _init_trust_system(self):
        """Initialize trust system components based on config."""
        trust_cfg = self.config.trust

        # Trust Dossier
        self.dossier = TrustDossier(self.work_dir, self.config.problem)

        # Variance Gate
        self.variance_gate = VarianceGate(
            n_evaluations=trust_cfg.n_evaluations,
            variance_threshold=trust_cfg.variance_threshold,
            require_gate=trust_cfg.require_variance_gate,
        )

        # Exploit Detector
        self.exploit_detector = ExploitDetector(
            check_timing=trust_cfg.check_timing_anomaly,
            timing_threshold_ms=trust_cfg.timing_anomaly_threshold_ms,
            check_output=trust_cfg.check_output_integrity,
            output_max_value=trust_cfg.output_max_value,
            check_determinism=trust_cfg.check_eval_determinism,
        )

        # Canary Test
        self.canary_test = CanaryTest(
            strict_mode=trust_cfg.canary_test_strict,
        )

        # Register validators
        if trust_cfg.extended_test_command:
            from .trust.validators import ExtendedTestValidator
            ValidatorRegistry.register(
                "extended",
                lambda: ExtendedTestValidator(trust_cfg.extended_test_command),
            )

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

        # Run canary test if enabled (verifies trust system works)
        if self.config.trust.canary_test_enabled:
            canary_passed = await self._run_canary_test()
            if not canary_passed and self.config.trust.canary_test_strict:
                raise RuntimeError(
                    "Canary test failed - trust system is not working properly. "
                    "Set canary_test_strict=false to continue anyway."
                )

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

        # Build parent map for adversary (maps output file -> parent solution)
        parent_map = {}
        for i, parent in enumerate(top_n):
            if i >= self.config.mutation_variants:
                break
            variant = chr(ord("a") + i)
            output_file = str(self.mutations_dir / f"gen{self.generation}{variant}.py")
            parent_map[output_file] = parent

        if valid_mutations:
            files_to_eval = [m["file"] for m in valid_mutations]
            evaluations = await self._spawn_evaluator(files_to_eval)

            # Run adversary challenge on evaluations (if trust system enabled)
            if self.config.trust.enabled:
                print("\n  [TRUST] Running adversary validation...")
                evaluations = await self._challenge_candidates(
                    candidates=valid_mutations,
                    parent_map=parent_map,
                    evaluations=evaluations,
                )

            # Match evaluations back to mutations and build results
            eval_by_file = {e.get("file"): e for e in evaluations}
            for m in valid_mutations:
                eval_result = eval_by_file.get(m["file"], {})
                fitness = eval_result.get("fitness", 0)
                is_valid = eval_result.get("valid", False)

                # Check adversary recommendation
                adversary_rejected = (
                    eval_result.get("adversary_reviewed", False)
                    and eval_result.get("adversary_recommendation") == "reject"
                )

                # Determine decision
                champion_fitness = old_champion.get("fitness", 0) if old_champion else 0
                if is_valid and fitness > 0 and not adversary_rejected:
                    decision = "KEEP"
                    # Add to population
                    combined = {**m, **eval_result}
                    self.population.append(combined)
                else:
                    decision = "DROP"

                # Build result entry with trust info
                result_entry = {
                    "variant": m.get("variant", "?"),
                    "mutation_type": m.get("mutation_type", "unknown"),
                    "fitness": fitness,
                    "decision": decision,
                    "file": m.get("file"),
                    "error": eval_result.get("error", "") if not is_valid else "",
                }

                # Add trust info if reviewed
                if eval_result.get("adversary_reviewed"):
                    result_entry["trust_score"] = eval_result.get("trust_score", 1.0)
                    result_entry["original_fitness"] = eval_result.get("original_fitness", fitness)
                    if adversary_rejected:
                        result_entry["error"] = "Adversary rejected"

                gen_results.append(result_entry)

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

    async def _spawn_adversary(
        self,
        candidate: dict,
        parent: dict | None,
        evaluation_result: dict,
    ) -> dict[str, Any]:
        """
        Spawn an adversary agent to challenge a candidate solution.

        The adversary reviews the candidate and returns:
        - trust_score: 0.0 to 1.0
        - escalation_level: 0-3
        - recommendation: accept/challenge/reject
        - adjusted_fitness: raw_fitness * trust_factor
        """
        population_best = self.champion.get("fitness", 0) if self.champion else 0

        prompt = get_adversary_prompt(
            candidate=candidate,
            parent=parent,
            evaluation_result=evaluation_result,
            mode=self.config.mode,
            generation=self.generation,
            population_best=population_best,
        )

        result = await self._run_agent(
            system=ADVERSARY_SYSTEM,
            prompt=prompt,
            tools=["Read", "Bash", "Glob"],
            max_turns=self.config.max_turns_per_agent,
        )

        parsed = self._parse_json_from_result(result)
        if parsed:
            return parsed

        # Default response if parsing fails
        return {
            "trust_score": 0.5,
            "escalation_level": 0,
            "flags": ["Could not parse adversary response"],
            "recommendation": "accept",
            "adjusted_fitness": evaluation_result.get("fitness", 0) * 0.5,
            "analysis": "Adversary response parsing failed, applying default trust penalty",
        }

    async def _run_escalation(
        self,
        candidate: dict,
        trust_result: dict,
        escalation_level: int,
    ) -> dict[str, Any]:
        """
        Run escalated validation for a candidate flagged by the adversary.

        Higher escalation levels run more comprehensive tests.
        """
        prompt = get_escalation_prompt(
            candidate=candidate,
            trust_result=trust_result,
            escalation_level=escalation_level,
            mode=self.config.mode,
            extended_test_command=self.config.trust.extended_test_command,
        )

        result = await self._run_agent(
            system=ADVERSARY_SYSTEM,  # Reuse adversary system for escalation
            prompt=prompt,
            tools=["Read", "Bash", "Glob"],
            max_turns=self.config.max_turns_per_agent + 5,  # Extra turns for thorough testing
        )

        parsed = self._parse_json_from_result(result)
        if parsed:
            return parsed

        # Default to rejection if parsing fails during escalation
        return {
            "escalation_passed": False,
            "revised_fitness": 0,
            "revised_trust": 0.0,
            "tests_run": [],
            "failures": ["Could not parse escalation response"],
            "final_recommendation": "reject",
        }

    async def _spawn_arbitrator(
        self,
        candidate_file: str,
        fitness: float,
        champion_fitness: float,
        trust_score: float,
        flags: list[str],
        adversary_analysis: str,
    ) -> str:
        """
        Spawn an arbitrator agent to provide balanced analysis for human escalation.

        The arbitrator reviews the situation neutrally and helps the human
        understand the tradeoffs of accepting vs rejecting the candidate.

        Returns:
            Formatted arbitrator analysis text for display to human.
        """
        # Try to read candidate code for context
        candidate_code = None
        try:
            with open(candidate_file, 'r') as f:
                candidate_code = f.read()
        except Exception:
            pass  # Code reading is optional

        prompt = get_arbitrator_prompt(
            candidate_file=candidate_file,
            fitness=fitness,
            champion_fitness=champion_fitness,
            trust_score=trust_score,
            flags=flags,
            adversary_analysis=adversary_analysis,
            mode=self.config.mode,
            generation=self.generation,
            candidate_code=candidate_code,
        )

        result = await self._run_agent(
            system=ARBITRATOR_SYSTEM,
            prompt=prompt,
            tools=["Read"],  # Read-only access for code review
            max_turns=5,  # Quick analysis, not deep investigation
        )

        return result

    def _ask_human_escalation(
        self,
        candidate_file: str,
        fitness: float,
        champion_fitness: float,
        trust_score: float,
        flags: list[str],
        analysis: str,
        arbitrator_analysis: str | None = None,
    ) -> dict[str, Any]:
        """
        Ask human to make final decision on borderline candidate.

        Args:
            candidate_file: Path to the candidate solution
            fitness: Candidate's fitness score
            champion_fitness: Current champion's fitness
            trust_score: Trust score from adversary
            flags: Flags raised by adversary
            analysis: Adversary's detailed analysis
            arbitrator_analysis: Optional neutral analysis from arbitrator agent

        Returns:
            dict with:
            - decision: "accept" | "reject" | "adjust"
            - adjusted_trust: float (if decision is "adjust")
        """
        import sys
        import select

        trust_cfg = self.config.trust

        # Build summary display
        print("\n" + "=" * 70)
        print("  HUMAN ESCALATION - Trust System Needs Your Input")
        print("=" * 70)
        print(f"\n  Candidate: {candidate_file}")
        print(f"  Fitness:   {fitness:.4f}  (current champion: {champion_fitness:.4f})")
        print(f"  Trust:     {trust_score:.2f}  (threshold: {trust_cfg.reject_threshold})")

        if fitness > champion_fitness:
            improvement = ((fitness - champion_fitness) / champion_fitness * 100) if champion_fitness > 0 else 100
            print(f"  Status:    POTENTIAL NEW CHAMPION (+{improvement:.1f}%)")
        else:
            print(f"  Status:    Below champion")

        if flags:
            print(f"\n  Flags ({len(flags)}):")
            for flag in flags[:5]:  # Show first 5 flags
                print(f"    - {flag[:60]}{'...' if len(flag) > 60 else ''}")
            if len(flags) > 5:
                print(f"    ... and {len(flags) - 5} more")

        # Show arbitrator analysis prominently if available
        if arbitrator_analysis:
            print("\n" + "-" * 70)
            print("  ARBITRATOR ANALYSIS (Neutral Assessment)")
            print("-" * 70)
            # Show the arbitrator's analysis - it's already formatted
            for line in arbitrator_analysis.split('\n'):
                print(f"  {line}")
            print("-" * 70)
        elif analysis:
            # Fall back to adversary analysis if no arbitrator
            print(f"\n  Adversary Analysis (truncated):")
            analysis_preview = analysis[:300] + "..." if len(analysis) > 300 else analysis
            for line in analysis_preview.split('\n')[:5]:
                print(f"    {line}")

        print("\n" + "-" * 70)
        print("  Options:")
        print("    [A] Accept - Trust this candidate (override rejection)")
        print("    [R] Reject - Confirm rejection (default)")
        print("    [T] Adjust Trust - Enter custom trust score (0.0-1.0)")
        print("    [V] View Full - Show adversary analysis, arbitrator analysis, and flags")
        print("-" * 70)

        # Handle timeout
        timeout = trust_cfg.human_escalation_timeout
        if timeout > 0:
            print(f"  (Auto-reject in {timeout}s if no response)")

        try:
            while True:
                if timeout > 0:
                    # Use select for timeout on Unix
                    if sys.platform != 'win32':
                        ready, _, _ = select.select([sys.stdin], [], [], timeout)
                        if not ready:
                            print("\n  [TIMEOUT] No response - rejecting candidate")
                            return {"decision": "reject", "reason": "timeout"}

                response = input("\n  Your choice [A/R/T/V]: ").strip().upper()

                if response == 'A':
                    print("  [ACCEPTED] Human override - accepting candidate")
                    return {"decision": "accept", "adjusted_trust": 0.8}

                elif response == 'R' or response == '':
                    print("  [REJECTED] Human confirmed rejection")
                    return {"decision": "reject", "reason": "human_confirmed"}

                elif response == 'T':
                    try:
                        new_trust = float(input("  Enter trust score (0.0-1.0): ").strip())
                        if 0.0 <= new_trust <= 1.0:
                            decision = "accept" if new_trust >= trust_cfg.reject_threshold else "reject"
                            print(f"  [ADJUSTED] Trust set to {new_trust:.2f} -> {decision}")
                            return {"decision": decision, "adjusted_trust": new_trust}
                        else:
                            print("  Invalid: must be between 0.0 and 1.0")
                    except ValueError:
                        print("  Invalid: enter a decimal number")

                elif response == 'V':
                    if arbitrator_analysis:
                        print("\n  === ARBITRATOR ANALYSIS ===")
                        print(arbitrator_analysis)
                    print("\n  === ADVERSARY ANALYSIS ===")
                    print(analysis if analysis else "(No adversary analysis available)")
                    print("\n  === FLAGS ===")
                    for flag in flags:
                        print(f"    - {flag}")
                    print()

                else:
                    print("  Invalid choice. Enter A, R, T, or V.")

        except (KeyboardInterrupt, EOFError):
            print("\n  [INTERRUPTED] Rejecting candidate")
            return {"decision": "reject", "reason": "interrupted"}

    async def _challenge_candidates(
        self,
        candidates: list[dict],
        parent_map: dict[str, dict],
        evaluations: list[dict],
    ) -> list[dict]:
        """
        Run adversary challenge on candidates that need trust verification.

        Includes:
        - Exploit detection (timing, output integrity)
        - Variance gates (if n_evaluations > 1)
        - Adversary agent review
        - Escalation ladder
        - Trust dossier recording

        Returns updated evaluations with trust scores and adjusted fitness.

        Args:
            candidates: List of candidate solutions with mutation info
            parent_map: Map from candidate file to parent solution
            evaluations: List of evaluator results with fitness scores
        """
        if not self.config.trust.enabled:
            # Trust system disabled - return evaluations unchanged
            return evaluations

        trust_cfg = self.config.trust
        updated_evals = []
        population_best = self.champion.get("fitness", 0) if self.champion else 0

        for eval_result in evaluations:
            file_path = eval_result.get("file", "")
            fitness = eval_result.get("fitness", 0)
            valid = eval_result.get("valid", False)
            eval_time_ms = eval_result.get("evaluation_time_ms")

            # Find the corresponding candidate
            candidate = next((c for c in candidates if c.get("file") == file_path), None)
            parent = parent_map.get(file_path)

            # Track all flags and checks
            all_flags = []
            exploit_results = {}
            variance_stats = None

            # === EXPLOIT DETECTION (B) ===
            # Run basic exploit checks before adversary review
            exploit_results = self.exploit_detector.run_all_checks(
                fitness=fitness,
                evaluation_time_ms=eval_time_ms,
            )
            exploit_flags = self.exploit_detector.get_flags(exploit_results)
            all_flags.extend(exploit_flags)

            # Critical exploit = immediate rejection
            if self.exploit_detector.has_critical(exploit_results):
                print(f"  [EXPLOIT] Critical exploit detected: {exploit_flags}")
                eval_result["trust_score"] = 0.0
                eval_result["adversary_reviewed"] = True
                eval_result["adversary_recommendation"] = "reject"
                eval_result["adversary_flags"] = exploit_flags
                eval_result["exploit_checks"] = {k: v.to_dict() for k, v in exploit_results.items()}
                eval_result["fitness"] = 0
                updated_evals.append(eval_result)

                # Record in dossier
                self.dossier.record_evaluation(
                    candidate_file=file_path,
                    generation=self.generation,
                    fitness=fitness,
                    trust_score=0.0,
                    recommendation="reject",
                    flags=exploit_flags,
                    exploit_checks={k: v.to_dict() for k, v in exploit_results.items()},
                    analysis="Critical exploit detected - immediate rejection",
                )
                continue

            # Check if this candidate needs adversary review
            needs_review = False

            if not valid or fitness <= 0:
                # Invalid solutions don't need adversary
                updated_evals.append(eval_result)
                continue

            # Check for suspicious fitness jump
            if parent and parent.get("fitness", 0) > 0:
                jump_pct = ((fitness - parent["fitness"]) / parent["fitness"]) * 100
                if jump_pct > trust_cfg.suspicious_jump_pct:
                    needs_review = True
                    all_flags.append(f"Suspicious jump: {jump_pct:.1f}%")
                    print(f"  [!] Suspicious jump: {jump_pct:.1f}% > {trust_cfg.suspicious_jump_pct}%")

            # Check if this would be a new champion
            if trust_cfg.require_adversary_for_champion and fitness > population_best:
                needs_review = True
                print(f"  [!] New champion candidate: {fitness:.4f} > {population_best:.4f}")

            # Check if exploit detection raised warnings
            if exploit_flags:
                needs_review = True
                print(f"  [!] Exploit warnings: {exploit_flags}")

            if not needs_review:
                # No review needed - keep original fitness
                eval_result["trust_score"] = 1.0
                eval_result["adversary_reviewed"] = False
                eval_result["exploit_checks"] = {k: v.to_dict() for k, v in exploit_results.items()}
                updated_evals.append(eval_result)
                continue

            # === VARIANCE GATES (A) ===
            # Run multiple evaluations if configured
            if trust_cfg.n_evaluations > 1:
                print(f"  [VARIANCE] Running {trust_cfg.n_evaluations} evaluations...")
                # Note: For now, we use the single fitness value
                # In a full implementation, this would re-run the evaluator N times
                # For demonstration, we simulate with the single value
                fitness_values = [fitness]  # TODO: Actually re-evaluate N times

                variance_stats = self.variance_gate.check_consistency(fitness_values)
                variance_flags = self.variance_gate.get_flags(variance_stats)
                all_flags.extend(variance_flags)

                if not variance_stats.passed and trust_cfg.require_variance_gate:
                    print(f"  [VARIANCE] FAILED: CV={variance_stats.cv:.4f} > {trust_cfg.variance_threshold}")
                    eval_result["trust_score"] = 0.0
                    eval_result["adversary_reviewed"] = True
                    eval_result["adversary_recommendation"] = "reject"
                    eval_result["adversary_flags"] = all_flags
                    eval_result["variance_stats"] = variance_stats.to_dict()
                    eval_result["fitness"] = 0
                    updated_evals.append(eval_result)

                    # Record in dossier
                    self.dossier.record_evaluation(
                        candidate_file=file_path,
                        generation=self.generation,
                        fitness=fitness,
                        trust_score=0.0,
                        recommendation="reject",
                        flags=all_flags,
                        variance_stats=variance_stats.to_dict(),
                        analysis="Failed variance gate",
                    )
                    continue

            # === ADVERSARY AGENT REVIEW ===
            print(f"  [ADVERSARY] Challenging: {file_path.split('/')[-1]}")
            trust_result = await self._spawn_adversary(
                candidate=candidate or {"file": file_path},
                parent=parent,
                evaluation_result=eval_result,
            )

            trust_score = trust_result.get("trust_score", 0.5)
            escalation_level = trust_result.get("escalation_level", 0)
            recommendation = trust_result.get("recommendation", "accept")
            adversary_flags = trust_result.get("flags", [])
            all_flags.extend(adversary_flags)

            print(f"  [ADVERSARY] Trust: {trust_score:.2f}, Rec: {recommendation}, Flags: {len(adversary_flags)}")

            # === ESCALATION LADDER (C) ===
            escalation_history = []
            if (
                recommendation == "challenge"
                and escalation_level > 0
                and escalation_level <= trust_cfg.max_escalation_level
            ):
                print(f"  [ESCALATION] Level {escalation_level} validation...")
                escalation_result = await self._run_escalation(
                    candidate=candidate or {"file": file_path},
                    trust_result=trust_result,
                    escalation_level=escalation_level,
                )

                escalation_history.append({
                    "level": escalation_level,
                    "passed": escalation_result.get("escalation_passed", False),
                    "tests_run": escalation_result.get("tests_run", []),
                    "failures": escalation_result.get("failures", []),
                })

                if escalation_result.get("escalation_passed"):
                    trust_score = escalation_result.get("revised_trust", trust_score)
                    fitness = escalation_result.get("revised_fitness", fitness)
                    recommendation = escalation_result.get("final_recommendation", "accept")
                    print(f"  [ESCALATION] Passed! Revised trust: {trust_score:.2f}")
                else:
                    recommendation = "reject"
                    trust_score = 0.0
                    all_flags.extend(escalation_result.get("failures", []))
                    print(f"  [ESCALATION] Failed: {escalation_result.get('failures', [])}")

            # Apply trust adjustment
            if recommendation == "reject" or trust_score < trust_cfg.reject_threshold:
                # === HUMAN-IN-THE-LOOP ESCALATION ===
                # Before rejecting, check if human should review borderline cases
                human_overrode = False

                if (
                    trust_cfg.human_escalation_enabled
                    and fitness > trust_cfg.human_escalation_min_fitness
                ):
                    # Check if this is a potential champion being rejected
                    is_potential_champion = fitness > population_best
                    is_borderline = trust_score >= (trust_cfg.reject_threshold * 0.5)  # Within 50% of threshold

                    should_escalate = (
                        (is_potential_champion and trust_cfg.human_escalation_on_champion_reject)
                        or (is_borderline and not is_potential_champion)
                    )

                    if should_escalate:
                        print(f"  [HUMAN] Escalating to human review...")

                        # Spawn arbitrator agent for neutral analysis
                        print(f"  [ARBITRATOR] Getting neutral assessment...")
                        arbitrator_analysis = await self._spawn_arbitrator(
                            candidate_file=file_path,
                            fitness=fitness,
                            champion_fitness=population_best,
                            trust_score=trust_score,
                            flags=all_flags,
                            adversary_analysis=trust_result.get("analysis", ""),
                        )

                        human_result = self._ask_human_escalation(
                            candidate_file=file_path,
                            fitness=fitness,
                            champion_fitness=population_best,
                            trust_score=trust_score,
                            flags=all_flags,
                            analysis=trust_result.get("analysis", ""),
                            arbitrator_analysis=arbitrator_analysis,
                        )

                        if human_result["decision"] == "accept":
                            human_overrode = True
                            # Use adjusted trust if provided, otherwise use a passing value
                            trust_score = human_result.get("adjusted_trust", trust_cfg.accept_threshold)
                            recommendation = "accept"
                            all_flags.append(f"human_override: accepted with trust {trust_score:.2f}")
                            print(f"  [HUMAN] Override accepted - trust adjusted to {trust_score:.2f}")
                        elif human_result["decision"] == "reject":
                            all_flags.append(f"human_confirmed_rejection: {human_result.get('reason', 'explicit')}")
                        # If decision is "adjust", trust_score was already set above

                        # Record in dossier
                        self.dossier.record_human_escalation(
                            candidate_file=file_path,
                            generation=self.generation,
                            fitness=fitness,
                            original_trust=eval_result.get("trust_score", trust_score),
                            decision=human_result["decision"],
                            adjusted_trust=human_result.get("adjusted_trust"),
                            reason=human_result.get("reason", ""),
                        )

                if not human_overrode:
                    adjusted_fitness = 0
                elif trust_cfg.apply_trust_adjustment:
                    adjusted_fitness = fitness * trust_score
                else:
                    adjusted_fitness = fitness
            elif trust_cfg.apply_trust_adjustment:
                adjusted_fitness = fitness * trust_score
            else:
                adjusted_fitness = fitness

            # Update evaluation with trust info
            eval_result["trust_score"] = trust_score
            eval_result["original_fitness"] = fitness
            eval_result["fitness"] = adjusted_fitness
            eval_result["adversary_reviewed"] = True
            eval_result["adversary_recommendation"] = recommendation
            eval_result["adversary_flags"] = all_flags
            eval_result["adversary_analysis"] = trust_result.get("analysis", "")
            eval_result["exploit_checks"] = {k: v.to_dict() for k, v in exploit_results.items()}
            if variance_stats:
                eval_result["variance_stats"] = variance_stats.to_dict()

            updated_evals.append(eval_result)

            # === TRUST DOSSIER (E) ===
            # Record in dossier for reporting
            self.dossier.record_evaluation(
                candidate_file=file_path,
                generation=self.generation,
                fitness=fitness,
                trust_score=trust_score,
                recommendation=recommendation,
                flags=all_flags,
                variance_stats=variance_stats.to_dict() if variance_stats else None,
                exploit_checks={k: v.to_dict() for k, v in exploit_results.items()},
                escalation_history=escalation_history,
                analysis=trust_result.get("analysis", ""),
            )

        return updated_evals

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

        # Generate trust dossier if enabled
        if self.config.trust.generate_dossier:
            self.dossier.save(include_history=self.config.trust.dossier_include_history)
            print(f"[TRUST] Dossier saved to: {self.dossier.dossier_path}")

        return {
            "problem": self.config.problem,
            "mode": self.config.mode,
            "generations": self.generation,
            "champion": self.champion,
            "population": self.population,
            "history": self.history,
        }

    async def _run_canary_test(self) -> bool:
        """
        Run canary test to verify trust system is working.

        Injects known-bad candidates and verifies they're rejected.

        Returns:
            True if all canaries were correctly rejected
        """
        print("\n[CANARY] Running trust system verification...")

        # Create canary directory
        canary_dir = self.work_dir / "canary_tests"
        canary_dir.mkdir(exist_ok=True)

        # Run canary tests using the default trust evaluator
        # The default evaluator is designed to catch all built-in canary types
        results = self.canary_test.run_all_canaries(
            output_dir=canary_dir,
            trust_evaluate_fn=None,  # Uses CanaryTest.default_trust_evaluator
            mode=self.config.mode,
        )

        # Record in dossier
        for result in results:
            self.dossier.record_canary_result(
                canary_fitness=result.canary_fitness,
                canary_trust=result.canary_trust,
                canary_rejected=result.canary_rejected,
                canary_flags=result.canary_flags,
            )

        # Report results
        passed = sum(1 for r in results if r.system_working)
        total = len(results)
        all_passed = self.canary_test.all_passed()

        if all_passed:
            print(f"[CANARY] PASSED: {passed}/{total} canaries correctly rejected")
        else:
            print(f"[CANARY] FAILED: Only {passed}/{total} canaries rejected")
            for failure in self.canary_test.get_failures():
                print(f"  - {failure.canary_type}: trust={failure.canary_trust:.2f}, rejected={failure.canary_rejected}")

        return all_passed

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
