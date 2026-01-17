"""
High-level Memory Manager API for evolve-sdk.

Provides typed, semantic access to project memory without raw JSON reading.
Supports both memvid (for large projects) and JSON (for small projects/fallback).

Usage:
    from evolve_sdk.memory import MemoryManager

    memory = MemoryManager(project_path)

    # Get all DRIs - ALWAYS call before submissions
    dris = memory.get_all_dris()

    # Semantic search for relevant DRIs
    relevant = memory.get_relevant_dris("submitting to huggingface")

    # Add a new DRI
    memory.add_dri(DRI(
        id="DRI-015",
        title="Always verify uploads",
        root_cause="Upload succeeded but files were corrupted",
    ))

    # Get recent submissions
    submissions = memory.get_recent_submissions(limit=5)

    # Add evaluation with sample size validation
    memory.add_evaluation(ModelEvaluation(
        model_name="chess-v6",
        metrics={"acpl": 78.1},
        sample_size=100,  # Enforced minimum
        methodology="Stockfish L20 on 100 positions",
        verdict="Competitive with baseline",
    ))
"""

import json
from pathlib import Path
from typing import Any

from .project_schemas import (
    DRI,
    Submission,
    ModelEvaluation,
    ConfigChange,
    SCHEMA_TYPES,
    from_dict as schema_from_dict,
)
from .store import EvolutionMemory, MemoryConfig, JSONBackend, MEMVID_AVAILABLE

if MEMVID_AVAILABLE:
    from .store import MemvidBackend


class MemoryManager:
    """
    High-level memory access API.

    Provides typed access to DRIs, submissions, evaluations, and config changes
    with semantic search capabilities. No more raw JSON reading.

    The manager automatically selects the appropriate backend:
    - memvid (.mv2) for large projects with semantic search
    - JSON for small projects or when memvid is unavailable
    """

    def __init__(
        self,
        project_path: Path | str,
        use_memvid: bool = True,
        problem_id: str = "",
    ):
        """
        Initialize the memory manager.

        Args:
            project_path: Path to the project directory (contains .evolve-sdk/)
            use_memvid: Whether to use memvid backend (falls back to JSON if unavailable)
            problem_id: Optional problem identifier for filtering
        """
        self.project_path = Path(project_path)
        self.problem_id = problem_id

        # Determine store path
        evolve_dir = self.project_path / ".evolve-sdk"
        if problem_id:
            store_dir = evolve_dir / problem_id
        else:
            store_dir = evolve_dir

        store_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend
        self._use_memvid = use_memvid and MEMVID_AVAILABLE
        if self._use_memvid:
            self._store_path = store_dir / "project_memory.mv2"
            try:
                self._backend = MemvidBackend(self._store_path)
            except Exception:
                # Fall back to JSON
                self._use_memvid = False
                self._store_path = store_dir / "project_memory.json"
                self._backend = JSONBackend(self._store_path)
        else:
            self._store_path = store_dir / "project_memory.json"
            self._backend = JSONBackend(self._store_path)

        # Cache for quick access
        self._cache: dict[str, list[dict]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load and categorize all frames into cache."""
        self._cache = {
            "DRI": [],
            "Submission": [],
            "ModelEvaluation": [],
            "ConfigChange": [],
        }

        try:
            frames = self._backend.load_all()
            for frame in frames:
                schema_type = frame.get("_schema_type")
                if schema_type in self._cache:
                    self._cache[schema_type].append(frame)
        except Exception:
            pass

    def _store(self, obj: DRI | Submission | ModelEvaluation | ConfigChange) -> str:
        """Store an object and update cache."""
        frame = obj.to_dict()
        frame_id = self._backend.store(frame)

        schema_type = frame.get("_schema_type")
        if schema_type in self._cache:
            self._cache[schema_type].append(frame)

        return frame_id

    # ==================== DRI Operations ====================

    def get_all_dris(self) -> list[DRI]:
        """
        Get all Don't Repeat Incidents.

        ALWAYS call this before any major action (submissions, uploads, etc.)
        to avoid repeating past mistakes.

        Returns:
            List of all DRIs, sorted by date (newest first)
        """
        dris = [DRI.from_dict(d) for d in self._cache.get("DRI", [])]
        return sorted(dris, key=lambda d: d.date, reverse=True)

    def get_relevant_dris(self, task_description: str, limit: int = 10) -> list[DRI]:
        """
        Semantic search for DRIs relevant to the current task.

        Uses vector similarity to find DRIs that might apply to what you're
        about to do, even if the exact keywords don't match.

        Args:
            task_description: Description of what you're about to do
            limit: Maximum number of results

        Returns:
            List of relevant DRIs, sorted by relevance

        Example:
            >>> memory.get_relevant_dris("uploading model to huggingface")
            [DRI-001: Template override broke model, DRI-003: Upload not verified, ...]
        """
        if self._use_memvid and hasattr(self._backend, "search"):
            results = self._backend.search(task_description, top_k=limit, frame_type="DRI")
            return [DRI.from_dict(r) for r in results]

        # Fallback: keyword matching
        keywords = set(task_description.lower().split())
        scored = []
        for dri_dict in self._cache.get("DRI", []):
            dri = DRI.from_dict(dri_dict)
            # Score based on keyword overlap
            text = f"{dri.title} {dri.description} {dri.root_cause} {' '.join(dri.tags)}".lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, dri))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [dri for _, dri in scored[:limit]]

    def get_dris_for_action(self, action_type: str) -> list[DRI]:
        """
        Get DRIs relevant to a specific action type.

        Args:
            action_type: Type of action (e.g., "huggingface_upload", "submission", "evaluation")

        Returns:
            List of DRIs that have this action in their related_actions
        """
        results = []
        for dri_dict in self._cache.get("DRI", []):
            dri = DRI.from_dict(dri_dict)
            if action_type in dri.related_actions or action_type in dri.tags:
                results.append(dri)
        return results

    def add_dri(self, dri: DRI) -> str:
        """
        Add a new Don't Repeat Incident.

        Args:
            dri: The DRI to add (must have id, title, and root_cause)

        Returns:
            Frame ID

        Raises:
            ValueError: If DRI is missing required fields
        """
        if not dri.id:
            raise ValueError("DRI must have an id")
        if not dri.title:
            raise ValueError("DRI must have a title")
        if not dri.root_cause:
            raise ValueError("DRI must have root_cause - document WHY it failed")

        return self._store(dri)

    # ==================== Submission Operations ====================

    def get_all_submissions(self) -> list[Submission]:
        """Get all submissions, sorted by date (newest first)."""
        submissions = [Submission.from_dict(d) for d in self._cache.get("Submission", [])]
        return sorted(submissions, key=lambda s: s.date, reverse=True)

    def get_recent_submissions(self, limit: int = 5) -> list[Submission]:
        """
        Get recent submissions for reference.

        Args:
            limit: Maximum number of submissions to return

        Returns:
            List of recent submissions, newest first
        """
        return self.get_all_submissions()[:limit]

    def get_successful_submissions(self) -> list[Submission]:
        """
        Get successful submissions to copy patterns from.

        Returns:
            List of submissions with status="succeeded"
        """
        return [s for s in self.get_all_submissions() if s.succeeded]

    def get_failed_submissions(self) -> list[Submission]:
        """
        Get failed submissions to learn from.

        Returns:
            List of submissions with status="failed"
        """
        return [s for s in self.get_all_submissions() if s.status == "failed"]

    def add_submission(self, submission: Submission) -> str:
        """
        Record a submission.

        Args:
            submission: The submission to record

        Returns:
            Frame ID
        """
        if not submission.id:
            raise ValueError("Submission must have an id")
        return self._store(submission)

    # ==================== Model Evaluation Operations ====================

    def get_all_evaluations(self) -> list[ModelEvaluation]:
        """Get all model evaluations, sorted by date (newest first)."""
        evals = [ModelEvaluation.from_dict(d) for d in self._cache.get("ModelEvaluation", [])]
        return sorted(evals, key=lambda e: e.eval_date, reverse=True)

    def get_model_evaluations(self, model_name: str | None = None) -> list[ModelEvaluation]:
        """
        Get model evaluations, optionally filtered by model name.

        Args:
            model_name: Optional model name to filter by

        Returns:
            List of evaluations matching the filter
        """
        evals = self.get_all_evaluations()
        if model_name:
            evals = [e for e in evals if model_name in e.model_name]
        return evals

    def add_evaluation(self, evaluation: ModelEvaluation) -> str:
        """
        Add a model evaluation with sample size validation.

        This enforces evidence requirements to prevent conclusions from
        insufficient data (see DRI-011).

        Args:
            evaluation: The evaluation to record

        Returns:
            Frame ID

        Raises:
            ValueError: If sample size is below minimum (100) or other validation fails
        """
        errors = evaluation.validate()
        if errors:
            raise ValueError(
                f"Evaluation failed validation:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return self._store(evaluation)

    # ==================== Config Change Operations ====================

    def get_config_changes(self, parameter: str | None = None) -> list[ConfigChange]:
        """
        Get config changes, optionally filtered by parameter name.

        Args:
            parameter: Optional parameter name to filter by

        Returns:
            List of config changes
        """
        changes = [ConfigChange.from_dict(d) for d in self._cache.get("ConfigChange", [])]
        if parameter:
            changes = [c for c in changes if parameter in c.parameter]
        return sorted(changes, key=lambda c: c.date, reverse=True)

    def add_config_change(self, change: ConfigChange) -> str:
        """
        Record a config change with evidence validation.

        Enforces evidence requirements to prevent making up config values
        without measurement (see DRI-002).

        Args:
            change: The config change to record

        Returns:
            Frame ID

        Raises:
            ValueError: If evidence or rationale is missing
        """
        errors = change.validate()
        if errors:
            raise ValueError(
                f"Config change failed validation:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return self._store(change)

    # ==================== Semantic Search ====================

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Semantic search across all memory types.

        Args:
            query: Natural language search query
            limit: Maximum results

        Returns:
            List of matching records with _search_score
        """
        if self._use_memvid and hasattr(self._backend, "search"):
            return self._backend.search(query, top_k=limit)

        # Fallback: keyword search across all types
        keywords = set(query.lower().split())
        results = []

        for schema_type, items in self._cache.items():
            for item in items:
                text = json.dumps(item).lower()
                score = sum(1 for kw in keywords if kw in text)
                if score > 0:
                    item_copy = item.copy()
                    item_copy["_search_score"] = score
                    results.append(item_copy)

        results.sort(key=lambda x: x.get("_search_score", 0), reverse=True)
        return results[:limit]

    # ==================== Pre-Action Checks ====================

    def pre_action_check(self, action_type: str, description: str = "") -> dict[str, Any]:
        """
        Run pre-action checks before a major operation.

        This should be called before submissions, uploads, evaluations, etc.
        Returns relevant DRIs and warnings.

        Args:
            action_type: Type of action being performed
            description: Optional description of the specific action

        Returns:
            Dictionary with:
                - relevant_dris: List of DRIs to review
                - warnings: List of warning messages
                - proceed: Boolean indicating if it's safe to proceed

        Example:
            >>> check = memory.pre_action_check("submission", "uploading v6 model to aicrowd")
            >>> if check["warnings"]:
            ...     print("Review these warnings before proceeding:")
            ...     for w in check["warnings"]:
            ...         print(f"  - {w}")
        """
        result = {
            "relevant_dris": [],
            "warnings": [],
            "proceed": True,
        }

        # Get DRIs for this action type
        action_dris = self.get_dris_for_action(action_type)
        result["relevant_dris"].extend(action_dris)

        # Get DRIs relevant to the description
        if description:
            desc_dris = self.get_relevant_dris(description, limit=5)
            for dri in desc_dris:
                if dri not in result["relevant_dris"]:
                    result["relevant_dris"].append(dri)

        # Generate warnings from DRIs
        for dri in result["relevant_dris"]:
            if dri.submissions_wasted > 0:
                result["warnings"].append(
                    f"{dri.id}: {dri.title} (wasted {dri.submissions_wasted} submissions)"
                )

        return result

    # ==================== Statistics ====================

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "store_path": str(self._store_path),
            "backend": "memvid" if self._use_memvid else "json",
            "counts": {
                "dris": len(self._cache.get("DRI", [])),
                "submissions": len(self._cache.get("Submission", [])),
                "evaluations": len(self._cache.get("ModelEvaluation", [])),
                "config_changes": len(self._cache.get("ConfigChange", [])),
            },
            "total_records": sum(len(v) for v in self._cache.values()),
        }

    def close(self):
        """Close the memory manager."""
        self._backend.close()


# Convenience function for quick access
def get_memory_manager(
    project_path: Path | str = ".",
    problem_id: str = "",
) -> MemoryManager:
    """
    Get a MemoryManager instance for the given project.

    Args:
        project_path: Path to project directory (default: current directory)
        problem_id: Optional problem identifier

    Returns:
        MemoryManager instance
    """
    return MemoryManager(project_path, problem_id=problem_id)
