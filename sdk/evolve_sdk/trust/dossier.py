"""Trust Dossier Generator - Creates markdown reports for trust decisions.

Generates comprehensive trust reports that document:
- Champion decisions and their justification
- Trust scores and variance analysis
- Exploit detection results
- Escalation history
- Multi-validator results
"""

from datetime import datetime
from pathlib import Path
from typing import Any


class TrustDossier:
    """Generates and manages trust dossier reports."""

    def __init__(self, work_dir: Path, problem: str):
        """
        Initialize the dossier generator.

        Args:
            work_dir: Evolution working directory
            problem: Problem description
        """
        self.work_dir = work_dir
        self.problem = problem
        self.entries: list[dict[str, Any]] = []
        self.dossier_path = work_dir / "trust_dossier.md"

    def record_evaluation(
        self,
        candidate_file: str,
        generation: int,
        fitness: float,
        trust_score: float,
        recommendation: str,
        flags: list[str] | None = None,
        variance_stats: dict[str, float] | None = None,
        exploit_checks: dict[str, Any] | None = None,
        escalation_history: list[dict] | None = None,
        validator_results: dict[str, Any] | None = None,
        analysis: str = "",
    ):
        """
        Record a trust evaluation for the dossier.

        Args:
            candidate_file: Path to the candidate solution
            generation: Generation number
            fitness: Raw fitness score
            trust_score: Trust score (0.0 to 1.0)
            recommendation: accept/challenge/reject
            flags: List of concerns raised
            variance_stats: Variance gate results {mean, std, cv, n_evals, passed}
            exploit_checks: Exploit detection results
            escalation_history: History of escalation decisions
            validator_results: Results from each validator
            analysis: Detailed analysis text
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "candidate_file": candidate_file,
            "generation": generation,
            "fitness": fitness,
            "trust_score": trust_score,
            "recommendation": recommendation,
            "flags": flags or [],
            "variance_stats": variance_stats,
            "exploit_checks": exploit_checks,
            "escalation_history": escalation_history or [],
            "validator_results": validator_results,
            "analysis": analysis,
        }
        self.entries.append(entry)

    def record_champion_decision(
        self,
        old_champion: dict | None,
        new_champion: dict,
        decision: str,  # "promoted" or "rejected"
        reason: str,
    ):
        """Record a champion promotion/rejection decision."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "champion_decision",
            "old_champion": old_champion,
            "new_champion": new_champion,
            "decision": decision,
            "reason": reason,
        }
        self.entries.append(entry)

    def record_canary_result(
        self,
        canary_fitness: float,
        canary_trust: float,
        canary_rejected: bool,
        canary_flags: list[str],
    ):
        """Record the canary test result."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "canary_test",
            "canary_fitness": canary_fitness,
            "canary_trust": canary_trust,
            "canary_rejected": canary_rejected,
            "canary_flags": canary_flags,
            "passed": canary_rejected,  # Canary should be rejected
        }
        self.entries.append(entry)

    def generate(self, include_history: bool = True) -> str:
        """
        Generate the markdown dossier.

        Args:
            include_history: Include full trust history

        Returns:
            Markdown content
        """
        lines = [
            f"# Trust Dossier",
            f"",
            f"> **Problem:** {self.problem}",
            f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
        ]

        # Summary statistics
        total_evals = len([e for e in self.entries if e.get("type") != "champion_decision" and e.get("type") != "canary_test"])
        rejections = len([e for e in self.entries if e.get("recommendation") == "reject"])
        acceptances = len([e for e in self.entries if e.get("recommendation") == "accept"])
        challenges = len([e for e in self.entries if e.get("recommendation") == "challenge"])

        lines.extend([
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Evaluations | {total_evals} |",
            f"| Accepted | {acceptances} |",
            f"| Challenged | {challenges} |",
            f"| Rejected | {rejections} |",
            f"",
        ])

        # Canary test results
        canary_entries = [e for e in self.entries if e.get("type") == "canary_test"]
        if canary_entries:
            lines.extend([
                f"## Canary Test",
                f"",
            ])
            for canary in canary_entries:
                status = "PASSED (canary rejected)" if canary.get("passed") else "FAILED (canary accepted!)"
                lines.extend([
                    f"- **Status:** {status}",
                    f"- **Canary Fitness:** {canary.get('canary_fitness', 0):.4f}",
                    f"- **Canary Trust:** {canary.get('canary_trust', 0):.2f}",
                    f"- **Flags:** {', '.join(canary.get('canary_flags', [])) or 'None'}",
                    f"",
                ])

        # Champion decisions
        champion_decisions = [e for e in self.entries if e.get("type") == "champion_decision"]
        if champion_decisions:
            lines.extend([
                f"## Champion Decisions",
                f"",
                f"| Time | Decision | Reason |",
                f"|------|----------|--------|",
            ])
            for cd in champion_decisions:
                time = cd.get("timestamp", "")[:19]
                decision = cd.get("decision", "")
                reason = cd.get("reason", "")[:50]
                lines.append(f"| {time} | {decision} | {reason} |")
            lines.append("")

        # Trust history
        if include_history:
            eval_entries = [e for e in self.entries if e.get("type") not in ("champion_decision", "canary_test")]
            if eval_entries:
                lines.extend([
                    f"## Trust History",
                    f"",
                ])
                for entry in eval_entries:
                    gen = entry.get("generation", "?")
                    candidate = entry.get("candidate_file", "").split("/")[-1]
                    fitness = entry.get("fitness", 0)
                    trust = entry.get("trust_score", 0)
                    rec = entry.get("recommendation", "?")
                    flags = entry.get("flags", [])

                    lines.extend([
                        f"### Gen {gen}: {candidate}",
                        f"",
                        f"- **Fitness:** {fitness:.4f}",
                        f"- **Trust Score:** {trust:.2f}",
                        f"- **Recommendation:** {rec.upper()}",
                    ])

                    if flags:
                        lines.append(f"- **Flags:**")
                        for flag in flags:
                            lines.append(f"  - {flag}")

                    # Variance stats
                    variance = entry.get("variance_stats")
                    if variance:
                        lines.extend([
                            f"- **Variance Analysis:**",
                            f"  - Mean: {variance.get('mean', 0):.4f}",
                            f"  - Std: {variance.get('std', 0):.4f}",
                            f"  - CV: {variance.get('cv', 0):.4f}",
                            f"  - N Evals: {variance.get('n_evals', 0)}",
                            f"  - Gate Passed: {'Yes' if variance.get('passed') else 'No'}",
                        ])

                    # Exploit checks
                    exploit = entry.get("exploit_checks")
                    if exploit:
                        lines.extend([
                            f"- **Exploit Detection:**",
                        ])
                        for check, result in exploit.items():
                            status = "PASS" if result.get("passed", True) else "FAIL"
                            lines.append(f"  - {check}: {status}")
                            if result.get("details"):
                                lines.append(f"    - {result['details']}")

                    # Analysis
                    analysis = entry.get("analysis")
                    if analysis:
                        lines.extend([
                            f"- **Analysis:** {analysis[:200]}{'...' if len(analysis) > 200 else ''}",
                        ])

                    lines.append("")

        # Footer
        lines.extend([
            f"---",
            f"",
            f"*Generated by [Agentic Evolve](https://github.com/anthropics/agentic-evolve) Trust System*",
        ])

        return "\n".join(lines)

    def save(self, include_history: bool = True):
        """Save the dossier to disk."""
        content = self.generate(include_history)
        self.dossier_path.write_text(content)

    def load(self) -> bool:
        """Load existing dossier entries (for resume). Returns True if loaded."""
        # For now, we don't persist entries - dossier is regenerated each run
        # Could add JSON persistence if needed
        return False
