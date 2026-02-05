"""
SWE-Refactor agent that uses Claude Code CLI to perform
Java refactoring tasks from the SWE-Refactor benchmark.

Adapted from RefactorBench's refactor_agent.py for Java/Maven/Gradle.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "SWE-Refactor" / "pure_refactoring_data.json"
REPOS_DIR = PROJECT_ROOT / "repos"
WORKDIRS_ROOT = PROJECT_ROOT / "workdirs"
RESULTS_DIR = PROJECT_ROOT / "results"

# JDK version mapping (requires jenv or similar)
JDK_VERSIONS = {
    "1.8": "1.8",
    "8": "1.8",
    "11": "11",
    "17": "17",
    "21": "21",
}

DEFAULT_MODEL = "opus"
MAX_TURNS = 20  # Java refactoring may need more turns


def load_tasks() -> list[dict]:
    """Load all refactoring tasks from the dataset."""
    with open(DATA_FILE) as f:
        return json.load(f)


def get_tasks_by_project(project_name: str) -> list[dict]:
    """Get all tasks for a specific project."""
    tasks = load_tasks()
    return [t for t in tasks if t["projectName"] == project_name]


def get_task_by_id(unique_id: str) -> dict | None:
    """Get a specific task by its unique ID."""
    tasks = load_tasks()
    for t in tasks:
        if t["uniqueId"] == unique_id:
            return t
    return None


def get_repo_url(project_name: str) -> str:
    """Get the GitHub URL for a project."""
    # Map project names to repos
    repo_map = {
        "checkstyle": "https://github.com/checkstyle/checkstyle.git",
        "commons-lang": "https://github.com/apache/commons-lang.git",
        "commons-io": "https://github.com/apache/commons-io.git",
        "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
        "hibernate-search": "https://github.com/hibernate/hibernate-search.git",
        "javaparser": "https://github.com/javaparser/javaparser.git",
        "junit4": "https://github.com/junit-team/junit4.git",
        "junit5": "https://github.com/junit-team/junit5.git",
        "mockito": "https://github.com/mockito/mockito.git",
        "pmd": "https://github.com/pmd/pmd.git",
        "guava": "https://github.com/google/guava.git",
        "gson": "https://github.com/google/gson.git",
        "zxing": "https://github.com/zxing/zxing.git",
        "jadx": "https://github.com/skylot/jadx.git",
        "shenyu": "https://github.com/apache/shenyu.git",
        "hertzbeat": "https://github.com/apache/hertzbeat.git",
        "shardingsphere-elasticjob": "https://github.com/apache/shardingsphere-elasticjob.git",
        "shiro": "https://github.com/apache/shiro.git",
    }
    return repo_map.get(project_name)


def clone_repo(project_name: str) -> Path:
    """Clone a repository if not already cloned."""
    repo_path = REPOS_DIR / project_name
    if repo_path.exists():
        return repo_path

    url = get_repo_url(project_name)
    if not url:
        raise ValueError(f"Unknown project: {project_name}")

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {project_name} (full history)...", file=sys.stderr)
    subprocess.run(
        ["git", "clone", url, str(repo_path)],
        check=True,
        capture_output=True,
    )
    return repo_path


def setup_workdir(task: dict, work_root: Path) -> Path:
    """Set up a working directory for a task."""
    project_name = task["projectName"]
    commit_id = task["commitId"]
    unique_id = task["uniqueId"]

    # Clone if needed
    repo_path = clone_repo(project_name)

    # Create work directory
    workdir = work_root / f"{project_name}__{unique_id[:20]}"
    if workdir.exists():
        shutil.rmtree(workdir)

    # Copy repo to workdir
    shutil.copytree(repo_path, workdir)

    # Checkout the specific commit (parent of the refactoring commit)
    subprocess.run(
        ["git", "checkout", f"{commit_id}^"],
        cwd=str(workdir),
        check=True,
        capture_output=True,
    )

    return workdir


def switch_jdk(version: str) -> dict:
    """Switch JDK version using Homebrew paths, jenv, or JAVA_HOME.

    Returns a modified environment dict with JAVA_HOME and PATH set correctly.
    """
    jdk = JDK_VERSIONS.get(str(version), str(version))

    # Homebrew paths for macOS (most reliable)
    homebrew_paths = {
        "1.8": "/opt/homebrew/opt/openjdk@8/libexec/openjdk.jdk/Contents/Home",
        "8": "/opt/homebrew/opt/openjdk@8/libexec/openjdk.jdk/Contents/Home",
        "11": "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home",
        "17": "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "21": "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    }

    # Start with current environment
    env = os.environ.copy()

    # Find the Java home
    java_home = None

    # Try Homebrew path first
    if jdk in homebrew_paths:
        candidate = homebrew_paths[jdk]
        if Path(candidate).exists():
            java_home = candidate

    # Fall back to any available Java if requested version not found
    if not java_home:
        for fallback in ["21", "17", "11"]:
            if fallback in homebrew_paths and Path(homebrew_paths[fallback]).exists():
                java_home = homebrew_paths[fallback]
                print(f"Warning: JDK {jdk} not found, using JDK {fallback}", file=sys.stderr)
                break

    if java_home:
        env["JAVA_HOME"] = java_home
        # Prepend Java bin to PATH, include /opt/homebrew/bin for mvn/gradle
        env["PATH"] = f"{java_home}/bin:/opt/homebrew/bin:{env.get('PATH', '')}"
    else:
        print(f"Warning: Could not find any Java installation", file=sys.stderr)

    return env


def run_compile(task: dict, workdir: Path) -> dict:
    """Compile the project using its build system."""
    compile_cmd = task.get("compileCommand", "")
    jdk_version = task.get("compileJDK", "11")

    # Switch JDK and get environment
    env = switch_jdk(jdk_version)

    if not compile_cmd:
        # Detect build system
        if (workdir / "pom.xml").exists():
            compile_cmd = "mvn clean compile -DskipTests"
        elif (workdir / "build.gradle").exists():
            # Use --no-build-cache to avoid remote cache issues
            compile_cmd = "./gradlew clean compileJava -x test --no-build-cache"
        else:
            return {"passed": False, "error": "Unknown build system"}
    else:
        # If provided compile command uses Gradle, add --no-build-cache
        if "gradlew" in compile_cmd and "--no-build-cache" not in compile_cmd:
            compile_cmd = compile_cmd + " --no-build-cache"
        # Simplify 'build' to 'compileJava' if present (avoid running tests)
        if "gradlew" in compile_cmd and " build" in compile_cmd:
            compile_cmd = compile_cmd.replace(" build", " compileJava -x test")

    try:
        result = subprocess.run(
            compile_cmd,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-5000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Compile timeout (600s)"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def run_tests(task: dict, workdir: Path) -> dict:
    """Run project tests."""
    compile_cmd = task.get("compileCommand", "")
    jdk_version = task.get("compileJDK", "11")

    switch_jdk(jdk_version)

    # Use the project's compile command which often includes tests
    if "package" in compile_cmd or "test" in compile_cmd:
        test_cmd = compile_cmd
    elif (workdir / "pom.xml").exists():
        test_cmd = "mvn test -Dmaven.javadoc.skip=true"
    elif (workdir / "build.gradle").exists():
        test_cmd = "./gradlew test"
    else:
        return {"passed": False, "error": "Unknown build system"}

    try:
        result = subprocess.run(
            test_cmd,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=900,  # 15 min for tests
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout[-10000:] if result.stdout else "",
            "stderr": result.stderr[-5000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Test timeout (900s)"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def build_agent_prompt(task: dict, workdir: Path) -> str:
    """Build the prompt for the Claude Code agent."""
    refactor_type = task["type"]
    description = task["description"]
    source_before = task["sourceCodeBeforeRefactoring"]
    file_path = task["filePathBefore"]
    jdk_version = task.get("compileJDK", "11")
    compile_cmd = task.get("compileCommand", "mvn clean compile -DskipTests")

    # Build the prompt
    prompt = f"""You are an expert Java developer performing a code refactoring task.

## Working Directory
{workdir.resolve()}

## Refactoring Type
{refactor_type}

## Task Description
{description}

## Source Code to Refactor
File: {file_path}

```java
{source_before}
```

## Instructions

1. Read the file at `{file_path}` to understand the full context
2. Perform the {refactor_type} refactoring as described
3. Update all references, imports, and call sites as needed
4. The refactoring must preserve the program's behavior

## Validation

After making changes, compile the project to verify correctness:

```bash
# Set JDK version
jenv local {jdk_version}

# Compile
{compile_cmd.replace(' package ', ' compile -DskipTests ') if 'package' in compile_cmd else compile_cmd}
```

If compilation fails, read the error messages carefully and fix the issues.

## Refactoring Guidelines

For {refactor_type}:
"""

    # Add type-specific guidelines
    if "Extract Method" in refactor_type:
        prompt += """
- Extract the specified code into a new method
- Choose appropriate parameter types and return type
- Update the original location to call the new method
- Add any necessary imports
"""
    elif "Move Method" in refactor_type:
        prompt += """
- Move the method to the target class
- Update visibility as needed (private -> public if accessed externally)
- Update all call sites to use the new location
- Add import statements where needed
- Consider whether the method should be static in the new location
"""
    elif "Inline Method" in refactor_type:
        prompt += """
- Replace all calls to the method with its body
- Handle parameter substitution correctly
- Remove the original method after inlining all calls
- Simplify any redundant code after inlining
"""
    elif "Move And Rename" in refactor_type:
        prompt += """
- Move the method to the target class with a new name
- Update all call sites with the new class and method name
- Add necessary imports
"""

    prompt += """

IMPORTANT: Make sure all changes compile successfully before finishing.
"""

    return prompt


def run_agent(task: dict, workdir: Path, model: str = None, verbose: bool = False) -> dict:
    """Run the refactoring agent using Claude Code CLI."""
    model = model or DEFAULT_MODEL

    prompt = build_agent_prompt(task, workdir)

    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        "--output-format", "json",
        "--max-turns", str(MAX_TURNS),
        prompt,
    ]

    if verbose:
        print(f"  Running claude agent in {workdir}...", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(workdir.resolve()),
            timeout=900,  # 15 min max
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"},
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "turns_used": MAX_TURNS,
            "error": "Agent timed out (900s)",
        }
    except Exception as e:
        return {
            "passed": False,
            "turns_used": 0,
            "error": f"Failed to run claude: {e}",
        }

    # Parse turns used
    turns_used = 0
    try:
        output = json.loads(result.stdout)
        if isinstance(output, dict):
            turns_used = output.get("num_turns", 0)
    except (json.JSONDecodeError, TypeError):
        pass

    # Compile to check if refactoring is valid
    compile_result = run_compile(task, workdir)

    if verbose:
        status = "COMPILED" if compile_result["passed"] else "COMPILE FAILED"
        print(f"  {status}", file=sys.stderr)

    return {
        "passed": compile_result["passed"],
        "turns_used": turns_used,
        "compile_result": compile_result,
    }


def run_single_task(unique_id: str, model: str = None, verbose: bool = False) -> dict:
    """Run a single refactoring task."""
    task = get_task_by_id(unique_id)
    if not task:
        return {"error": f"Task not found: {unique_id}"}

    WORKDIRS_ROOT.mkdir(parents=True, exist_ok=True)
    workdir = setup_workdir(task, WORKDIRS_ROOT)

    result = run_agent(task, workdir, model=model, verbose=verbose)
    result["task_id"] = unique_id
    result["project"] = task["projectName"]
    result["type"] = task["type"]

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SWE-Refactor agent on a single task")
    parser.add_argument("--task-id", help="Unique task ID")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list-projects", action="store_true", help="List available projects")
    parser.add_argument("--list-tasks", help="List tasks for a project")
    args = parser.parse_args()

    if args.list_projects:
        tasks = load_tasks()
        projects = set(t["projectName"] for t in tasks)
        for p in sorted(projects):
            count = sum(1 for t in tasks if t["projectName"] == p)
            print(f"{p}: {count} tasks")
        sys.exit(0)

    if args.list_tasks:
        tasks = get_tasks_by_project(args.list_tasks)
        for t in tasks[:20]:  # Show first 20
            print(f"{t['uniqueId'][:40]}... - {t['type']}")
        if len(tasks) > 20:
            print(f"... and {len(tasks) - 20} more")
        sys.exit(0)

    if not args.task_id:
        parser.error("--task-id is required unless using --list-projects or --list-tasks")

    result = run_single_task(args.task_id, model=args.model, verbose=args.verbose)
    print(json.dumps(result, indent=2))
