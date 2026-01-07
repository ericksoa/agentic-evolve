"""Skill loader - reads evolution instructions from skill markdown files.

The SDK uses the same skill definitions as the /evolve CLI commands.
This ensures consistency between interactive and programmatic evolution.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SkillDefinition:
    """Parsed skill definition from markdown file."""

    name: str
    description: str
    mode: str  # "size", "perf", "ml"
    full_content: str

    # Extracted sections
    evaluation_contract: str
    fitness_function: str
    mutation_strategies: str
    crossover_strategies: str
    acceptance_criteria: str
    stopping_criteria: str
    directory_structure: str

    @property
    def system_prompt(self) -> str:
        """Generate system prompt for agents from skill content."""
        return f"""You are an evolution agent for {self.mode} optimization.

{self.description}

## Evaluation Contract
{self.evaluation_contract}

## Fitness Function
{self.fitness_function}

## Acceptance Criteria
{self.acceptance_criteria}
"""

    @property
    def initializer_guidance(self) -> str:
        """Get guidance for initializer agent."""
        return f"""## Mode: {self.mode}

{self.description}

Create diverse initial solutions exploring different algorithm families.

{self.mutation_strategies}
"""

    @property
    def mutator_guidance(self) -> str:
        """Get guidance for mutator agent."""
        return f"""## Mode: {self.mode}

Apply mutations to improve the solution.

{self.mutation_strategies}

{self.fitness_function}
"""

    @property
    def crossover_guidance(self) -> str:
        """Get guidance for crossover agent."""
        return f"""## Mode: {self.mode}

Combine innovations from multiple parent solutions.

{self.crossover_strategies}
"""

    @property
    def evaluator_guidance(self) -> str:
        """Get guidance for evaluator agent."""
        return f"""## Mode: {self.mode}

{self.evaluation_contract}

{self.fitness_function}

{self.acceptance_criteria}
"""


def extract_section(content: str, header: str, next_headers: list[str] = None) -> str:
    """Extract a section from markdown content by header."""
    # Look for the header
    header_patterns = [
        rf"^##+ {re.escape(header)}\s*$",
        rf"^##+ {re.escape(header)}[^\n]*$",
    ]

    for pattern in header_patterns:
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            start = match.end()
            break
    else:
        return ""

    # Find the end (next header of same or higher level)
    header_level = content[match.start():match.end()].count('#')
    end_pattern = rf"^#{{1,{header_level}}} "

    end_match = re.search(end_pattern, content[start:], re.MULTILINE)
    if end_match:
        end = start + end_match.start()
    else:
        end = len(content)

    return content[start:end].strip()


def parse_skill_file(path: Path) -> Optional[SkillDefinition]:
    """Parse a skill markdown file into a SkillDefinition."""
    if not path.exists():
        return None

    content = path.read_text()

    # Extract frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        content_body = content[frontmatter_match.end():]
    else:
        frontmatter = ""
        content_body = content

    # Parse description from frontmatter
    desc_match = re.search(r'description:\s*(.+)', frontmatter)
    description = desc_match.group(1).strip() if desc_match else ""

    # Determine mode from filename
    name = path.stem
    if "size" in name:
        mode = "size"
    elif "perf" in name:
        mode = "perf"
    elif "ml" in name:
        mode = "ml"
    else:
        mode = "size"  # Default

    # Extract key sections
    evaluation_contract = (
        extract_section(content_body, "Evaluation Contract") or
        extract_section(content_body, "Fitness Function")
    )

    fitness_function = (
        extract_section(content_body, "Fitness Function") or
        extract_section(content_body, "Scoring Formula")
    )

    mutation_strategies = (
        extract_section(content_body, "Mutation Operators") or
        extract_section(content_body, "Mutation Strategies") or
        extract_section(content_body, "Stage 3: Genetic Search")
    )

    crossover_strategies = (
        extract_section(content_body, "Crossover Operators") or
        extract_section(content_body, "Crossover Strategies") or
        extract_section(content_body, "Crossover Requirements")
    )

    acceptance_criteria = (
        extract_section(content_body, "Acceptance Criteria") or
        extract_section(content_body, "Acceptance Criteria (To Keep a Candidate)")
    )

    stopping_criteria = (
        extract_section(content_body, "Stopping Criteria") or
        extract_section(content_body, "Adaptive Stopping Criteria")
    )

    directory_structure = extract_section(content_body, "Directory Structure")

    return SkillDefinition(
        name=name,
        description=description,
        mode=mode,
        full_content=content_body,
        evaluation_contract=evaluation_contract,
        fitness_function=fitness_function,
        mutation_strategies=mutation_strategies,
        crossover_strategies=crossover_strategies,
        acceptance_criteria=acceptance_criteria,
        stopping_criteria=stopping_criteria,
        directory_structure=directory_structure,
    )


class SkillLoader:
    """Loads and caches skill definitions from markdown files."""

    # Default locations to search for skill files
    DEFAULT_SKILL_PATHS = [
        ".claude/commands",  # Project-level skills
        "../.claude/commands",  # Parent directory
        "~/.claude/commands",  # User-level skills
    ]

    def __init__(self, skill_paths: list[str] = None):
        """Initialize skill loader.

        Args:
            skill_paths: List of directories to search for skill files.
                        If None, uses DEFAULT_SKILL_PATHS.
        """
        self.skill_paths = skill_paths or self.DEFAULT_SKILL_PATHS
        self._cache: Dict[str, SkillDefinition] = {}

    def find_skill_file(self, mode: str) -> Optional[Path]:
        """Find the skill file for a given mode."""
        filename = f"evolve-{mode}.md"

        for skill_dir in self.skill_paths:
            path = Path(skill_dir).expanduser() / filename
            if path.exists():
                return path

        return None

    def load_skill(self, mode: str) -> Optional[SkillDefinition]:
        """Load and parse skill definition for a mode."""
        if mode in self._cache:
            return self._cache[mode]

        path = self.find_skill_file(mode)
        if path is None:
            return None

        skill = parse_skill_file(path)
        if skill:
            self._cache[mode] = skill

        return skill

    def get_all_skills(self) -> Dict[str, SkillDefinition]:
        """Load all available skills."""
        skills = {}
        for mode in ["size", "perf", "ml"]:
            skill = self.load_skill(mode)
            if skill:
                skills[mode] = skill
        return skills


# Global skill loader instance
_skill_loader: Optional[SkillLoader] = None


def get_skill_loader(skill_paths: list[str] = None) -> SkillLoader:
    """Get or create the global skill loader."""
    global _skill_loader
    if _skill_loader is None or skill_paths is not None:
        _skill_loader = SkillLoader(skill_paths)
    return _skill_loader


def get_skill(mode: str) -> Optional[SkillDefinition]:
    """Convenience function to get a skill definition."""
    return get_skill_loader().load_skill(mode)


def get_mode_guidance(mode: str, agent_type: str) -> str:
    """Get guidance for a specific agent type and mode.

    Args:
        mode: Evolution mode ("size", "perf", "ml")
        agent_type: One of "initializer", "mutator", "crossover", "evaluator"

    Returns:
        Guidance string from the skill definition, or empty string if not found.
    """
    skill = get_skill(mode)
    if skill is None:
        return ""

    guidance_map = {
        "initializer": skill.initializer_guidance,
        "mutator": skill.mutator_guidance,
        "crossover": skill.crossover_guidance,
        "evaluator": skill.evaluator_guidance,
    }

    return guidance_map.get(agent_type, "")
