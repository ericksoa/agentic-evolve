"""Pluggable Validator Interface for Escalation Ladder.

Validators provide independent validation at different escalation levels.
The system supports a chain of validators that can be configured per project.

Built-in validators:
- DefaultValidator: Uses the standard evaluation command
- ExtendedTestValidator: Runs extended test suite
- MultiCircuitValidator: Tests on multiple scenarios (like F1 circuits)

Custom validators can be registered for domain-specific validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path


@dataclass
class ValidationResult:
    """Result from a validator."""

    validator_name: str
    passed: bool
    fitness: float
    confidence: float
    details: str
    tests_run: list[str]
    failures: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "validator_name": self.validator_name,
            "passed": self.passed,
            "fitness": self.fitness,
            "confidence": self.confidence,
            "details": self.details,
            "tests_run": self.tests_run,
            "failures": self.failures,
            "metadata": self.metadata,
        }


class Validator(ABC):
    """
    Abstract base class for validators.

    Validators are called during escalation to provide independent
    verification of candidate solutions.
    """

    name: str = "base"
    escalation_level: int = 0  # 0=basic, 1=extended, 2=OOD, 3=gold

    @abstractmethod
    async def validate(
        self,
        candidate_file: str,
        mode: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a candidate solution.

        Args:
            candidate_file: Path to the solution file
            mode: Optimization mode (size, perf, ml)
            context: Additional context (parent, evaluation result, etc.)

        Returns:
            ValidationResult
        """
        pass

    def can_handle(self, mode: str) -> bool:
        """Check if this validator can handle the given mode."""
        return True  # Override in subclasses for mode-specific validators


class DefaultValidator(Validator):
    """
    Default validator using the standard evaluation command.

    This is the Level 0 validator - basic evaluation.
    """

    name = "default"
    escalation_level = 0

    def __init__(self, test_command: str | None = None):
        """
        Initialize with optional test command.

        Args:
            test_command: Command to run for evaluation (e.g., "python evaluate.py {solution}")
        """
        self.test_command = test_command

    async def validate(
        self,
        candidate_file: str,
        mode: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Run default validation."""
        import asyncio
        import subprocess

        tests_run = []
        failures = []

        if not self.test_command:
            # No test command - just check file exists
            if Path(candidate_file).exists():
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    fitness=context.get("fitness", 0),
                    confidence=1.0,
                    details="File exists (no test command configured)",
                    tests_run=["file_exists"],
                    failures=[],
                    metadata={},
                )
            else:
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    fitness=0,
                    confidence=0,
                    details="File does not exist",
                    tests_run=["file_exists"],
                    failures=["File not found"],
                    metadata={},
                )

        # Run test command
        cmd = self.test_command.replace("{solution}", candidate_file)
        tests_run.append(cmd)

        try:
            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=60)

            if result.returncode == 0:
                # Try to parse fitness from output
                output = stdout.decode()
                fitness = self._parse_fitness(output)
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    fitness=fitness,
                    confidence=1.0,
                    details=output[:500],
                    tests_run=tests_run,
                    failures=[],
                    metadata={"returncode": 0},
                )
            else:
                failures.append(f"Command failed: {stderr.decode()[:200]}")
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    fitness=0,
                    confidence=0,
                    details=f"Test command failed with code {result.returncode}",
                    tests_run=tests_run,
                    failures=failures,
                    metadata={"returncode": result.returncode},
                )

        except asyncio.TimeoutError:
            failures.append("Test command timed out")
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                fitness=0,
                confidence=0,
                details="Test command timed out after 60s",
                tests_run=tests_run,
                failures=failures,
                metadata={"timeout": True},
            )
        except Exception as e:
            failures.append(str(e))
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                fitness=0,
                confidence=0,
                details=f"Test command error: {e}",
                tests_run=tests_run,
                failures=failures,
                metadata={"error": str(e)},
            )

    def _parse_fitness(self, output: str) -> float:
        """Try to parse fitness from command output."""
        import re
        import json

        # Try JSON
        try:
            data = json.loads(output)
            if "fitness" in data:
                return float(data["fitness"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try regex patterns
        patterns = [
            r"fitness[:\s]+([0-9.]+)",
            r"score[:\s]+([0-9.]+)",
            r"result[:\s]+([0-9.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return 0.0


class ExtendedTestValidator(Validator):
    """
    Extended test validator for Level 1 escalation.

    Runs additional edge cases and boundary conditions.
    """

    name = "extended"
    escalation_level = 1

    def __init__(self, extended_command: str | None = None):
        """
        Initialize with extended test command.

        Args:
            extended_command: Command for extended testing
        """
        self.extended_command = extended_command

    async def validate(
        self,
        candidate_file: str,
        mode: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Run extended validation."""
        import asyncio

        tests_run = []
        failures = []

        if not self.extended_command:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                fitness=context.get("fitness", 0),
                confidence=0.8,
                details="No extended test command configured",
                tests_run=[],
                failures=[],
                metadata={"skipped": True},
            )

        cmd = self.extended_command.replace("{solution}", candidate_file)
        tests_run.append(cmd)

        try:
            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)

            if result.returncode == 0:
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    fitness=context.get("fitness", 0),
                    confidence=0.9,
                    details=stdout.decode()[:500],
                    tests_run=tests_run,
                    failures=[],
                    metadata={"returncode": 0},
                )
            else:
                failures.append(stderr.decode()[:200])
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    fitness=0,
                    confidence=0,
                    details="Extended tests failed",
                    tests_run=tests_run,
                    failures=failures,
                    metadata={"returncode": result.returncode},
                )

        except asyncio.TimeoutError:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                fitness=0,
                confidence=0,
                details="Extended test timed out",
                tests_run=tests_run,
                failures=["Timeout"],
                metadata={"timeout": True},
            )
        except Exception as e:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                fitness=0,
                confidence=0,
                details=f"Error: {e}",
                tests_run=tests_run,
                failures=[str(e)],
                metadata={"error": str(e)},
            )


class ValidatorRegistry:
    """
    Registry for validators.

    Manages available validators and their escalation levels.
    """

    _validators: dict[str, type[Validator]] = {}
    _instances: dict[str, Validator] = {}

    @classmethod
    def register(cls, name: str, validator_class: type[Validator]):
        """Register a validator class."""
        cls._validators[name] = validator_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Validator:
        """
        Get or create a validator instance.

        Args:
            name: Validator name
            **kwargs: Arguments to pass to validator constructor

        Returns:
            Validator instance
        """
        if name in cls._instances:
            return cls._instances[name]

        if name not in cls._validators:
            # Fall back to default
            validator = DefaultValidator(**kwargs)
        else:
            validator = cls._validators[name](**kwargs)

        cls._instances[name] = validator
        return validator

    @classmethod
    def get_chain(cls, names: list[str], **kwargs) -> list[Validator]:
        """
        Get a chain of validators in order.

        Args:
            names: List of validator names
            **kwargs: Arguments for all validators

        Returns:
            List of validator instances
        """
        return [cls.get(name, **kwargs) for name in names]

    @classmethod
    def get_for_level(cls, level: int) -> list[Validator]:
        """
        Get all validators at or below the given escalation level.

        Args:
            level: Maximum escalation level

        Returns:
            List of validators sorted by escalation level
        """
        validators = list(cls._instances.values())
        return sorted(
            [v for v in validators if v.escalation_level <= level],
            key=lambda v: v.escalation_level,
        )


# Register built-in validators
ValidatorRegistry.register("default", DefaultValidator)
ValidatorRegistry.register("extended", ExtendedTestValidator)
