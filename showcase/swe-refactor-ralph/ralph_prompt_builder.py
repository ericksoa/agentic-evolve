"""
Prompt builder for RALPH loop iterations on SWE-Refactor tasks.

Constructs iteration-aware prompts:
- Iteration 1 (cold start): Task description + instructions
- Iteration 2+ (warm start): Previous compilation output, errors,
  failed approaches from progress.json

Adapted from RefactorBench ralph_prompt_builder.py for Java projects.
"""

import json
from pathlib import Path


def build_prompt(
    task: dict,
    workdir: Path,
    progress: dict | None = None,
    iteration: int = 1,
    max_iterations: int = 10,
) -> str:
    """Build a prompt for a RALPH loop iteration.

    Args:
        task: SWE-Refactor task dict
        workdir: Working directory (already contains the repo)
        progress: progress.json contents (None for iteration 1)
        iteration: Current iteration number (1-based)
        max_iterations: Maximum iterations in this chain

    Returns:
        Complete prompt string for claude -p
    """
    parts = []

    # System instructions (always included)
    parts.append(_system_instructions())

    # Task description (always included)
    parts.append(_task_section(task, workdir))

    # Iteration context (iteration 2+)
    if iteration > 1 and progress:
        parts.append(_iteration_context(progress, iteration, max_iterations))

    # Validation instructions (always included)
    parts.append(_validation_section(task, workdir))

    return "\n\n".join(parts)


def _system_instructions() -> str:
    """Core system prompt for the Java refactoring agent."""
    return """You are an expert Java developer performing code refactoring tasks.
You must make precise, targeted changes to the codebase.

## Core Principles

1. **Read before writing.** Always read a file before modifying it. Understand the full context.
2. **Search exhaustively.** Use Grep and Glob to find ALL references to classes, methods, imports, and variables you are modifying. Missing a reference is the #1 cause of compilation failures.
3. **Modify ALL call sites.** When you rename, move, or change a method's signature, update EVERY file that imports or calls it.
4. **Preserve semantics.** Refactoring changes structure, not behavior. The code must do the same thing after your changes.
5. **Think about imports.** When moving code between files, update both the source file (exports) and all importing files. In Java, this means updating import statements.

## Strategy

1. Read the task description carefully and identify what needs to change
2. Use Grep to find ALL files that reference the things being changed
3. Plan your changes to cover every reference
4. Make all changes
5. Run the compilation command to check your work
6. If compilation fails, read the error output carefully and fix the issues"""


def _task_section(task: dict, workdir: Path) -> str:
    """Task description section."""
    refactor_type = task["type"]
    description = task["description"]
    source_before = task["sourceCodeBeforeRefactoring"]
    file_path = task["filePathBefore"]

    return f"""## Refactoring Task

You are working in the directory: {workdir.resolve()}

### Refactoring Type
{refactor_type}

### Task Description
{description}

### Source Code to Refactor
File: {file_path}

```java
{source_before}
```

### Refactoring Guidelines

{_get_type_guidelines(refactor_type)}"""


def _get_type_guidelines(refactor_type: str) -> str:
    """Get specific guidelines based on refactoring type."""
    if "Extract Method" in refactor_type and "Move" not in refactor_type:
        return """For Extract Method:
- Extract the specified code into a new method
- Choose appropriate parameter types and return type
- Update the original location to call the new method
- Add any necessary imports
- Ensure local variables used in the extracted code are passed as parameters"""

    elif "Move Method" in refactor_type and "Inline" not in refactor_type and "Rename" not in refactor_type:
        return """For Move Method:
- Move the method to the target class
- Update visibility as needed (private -> public if accessed externally)
- Update all call sites to use the new location
- Add import statements where needed
- Consider whether the method should be static in the new location
- If the method references instance fields, you may need to pass `this` or the relevant object"""

    elif "Inline Method" in refactor_type and "Move" not in refactor_type:
        return """For Inline Method:
- Replace all calls to the method with its body
- Handle parameter substitution correctly
- Remove the original method after inlining all calls
- Simplify any redundant code after inlining
- Be careful with control flow (return statements, exceptions)"""

    elif "Extract And Move Method" in refactor_type:
        return """For Extract And Move Method:
- First, extract the specified code into a new method
- Then, move that new method to the target class
- Update all call sites to use the new location
- Add necessary imports in all affected files
- This is a compound refactoring - do both steps correctly"""

    elif "Move And Rename" in refactor_type:
        return """For Move And Rename Method:
- Move the method to the target class with a new name
- Update all call sites with the new class and method name
- Add necessary imports
- Search for BOTH the old class and old method name to find all references"""

    elif "Move And Inline" in refactor_type:
        return """For Move And Inline Method:
- This is the most complex compound refactoring
- First understand where the method needs to move
- Inline the method body at all call sites in the target context
- Remove the original method
- Update all imports appropriately
- Be very careful with this one - search exhaustively for all usages"""

    else:
        return f"""For {refactor_type}:
- Carefully follow the task description
- Search for all references before making changes
- Update imports in all affected files
- Ensure compilation succeeds after changes"""


def _iteration_context(progress: dict, iteration: int, max_iterations: int) -> str:
    """Context from previous iterations (warm start)."""
    parts = []

    parts.append(f"## Iteration {iteration}/{max_iterations} - Previous Attempts")
    parts.append(
        "This is NOT your first attempt. Previous iterations have already modified "
        "files in this directory. Read the current state of files before making changes."
    )

    # Compile results from previous iterations
    compile_results = progress.get("compile_results", [])
    if compile_results:
        last = compile_results[-1]
        last_compiled = last.get("passed", False)
        parts.append(
            f"\n**Last iteration result:** {'COMPILED' if last_compiled else 'FAILED'}"
        )

        # Error output
        stderr = last.get("stderr", "")
        if stderr and not last_compiled:
            # Show last 2000 chars of compilation errors
            truncated = stderr[-2000:]
            if len(stderr) > 2000:
                truncated = "... (truncated)\n" + truncated
            parts.append(f"\n**Compilation errors:**\n```\n{truncated}\n```")

    # Failed approaches (DO NOT REPEAT)
    failed = progress.get("failed_approaches", [])
    if failed:
        parts.append("\n## DO NOT REPEAT These Failed Approaches")
        parts.append(
            "The following approaches were already tried and FAILED. "
            "Do NOT try them again. Find a different approach."
        )
        for i, approach in enumerate(failed[-10:], 1):  # Last 10
            parts.append(f"{i}. {approach}")

    parts.append("\n## Instructions for This Iteration")
    parts.append(
        "1. Read the CURRENT state of files (they may have been partially modified)\n"
        "2. Analyze the compilation errors above\n"
        "3. Try a DIFFERENT approach than the failed ones listed\n"
        "4. Fix ALL compilation errors\n"
        "5. Compile to verify your fix works"
    )

    return "\n".join(parts)


def _validation_section(task: dict, workdir: Path) -> str:
    """Validation/compile instructions."""
    jdk_version = task.get("compileJDK", "11")
    compile_cmd = task.get("compileCommand", "")

    if not compile_cmd:
        if (workdir / "pom.xml").exists():
            compile_cmd = "mvn clean compile -DskipTests"
        elif (workdir / "build.gradle").exists():
            compile_cmd = "./gradlew clean compileJava -x test"
        else:
            compile_cmd = "mvn clean compile -DskipTests"

    # Simplify compile command (remove package/test, ensure compile only)
    if "package" in compile_cmd and "compile" not in compile_cmd:
        compile_cmd = compile_cmd.replace("package", "compile -DskipTests")

    return f"""## Validation

After making all changes, compile the project to verify correctness.

### Compile Command

```bash
# Ensure correct JDK is used (JDK {jdk_version})
{compile_cmd}
```

### Important Notes

- The project must compile successfully for the refactoring to be considered correct
- Compilation errors indicate missing imports, incorrect method signatures, or unupdated call sites
- If you see "cannot find symbol" errors, search for ALL usages of that symbol
- If you see "package does not exist" errors, check import statements

IMPORTANT: Make ALL required code changes FIRST, then compile to validate. If compilation fails, read the error output carefully and fix the issues."""
