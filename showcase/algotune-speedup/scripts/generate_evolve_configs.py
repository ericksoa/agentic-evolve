#!/usr/bin/env python3
"""
Generate per-task evolve_config.json files from template.

Reads the template, substitutes task-specific values, and writes
one config file per task in configs/.

Usage:
    python scripts/generate_evolve_configs.py
    python scripts/generate_evolve_configs.py --subset configs/subset.json
"""

import json
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SHOWCASE_DIR / "evolve_config_template.json"
CONFIGS_DIR = SHOWCASE_DIR / "configs"
SOLUTIONS_DIR = SHOWCASE_DIR / "solutions"
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"


def load_template() -> str:
    """Load the config template as raw text for placeholder substitution."""
    return TEMPLATE_PATH.read_text()


def generate_config(task_name: str, template_text: str) -> dict:
    """Generate a task-specific config from the template."""
    # Substitute placeholders
    config_text = template_text.replace("{task_name}", task_name)
    config_text = config_text.replace("{showcase_dir}", str(SHOWCASE_DIR))
    config_text = config_text.replace("{solutions_dir}", str(SOLUTIONS_DIR))
    config_text = config_text.replace("{algotune_dir}", str(ALGOTUNE_DIR))

    config = json.loads(config_text)

    # Set working directory for evolve-sdk
    config["cwd"] = str(SHOWCASE_DIR)

    return config


def load_task_list(subset_path: Path = None) -> list:
    """Load task list from subset.json or generate default."""
    if subset_path and subset_path.exists():
        data = json.loads(subset_path.read_text())
        return data["tasks"]

    # Default subset file
    default_subset = CONFIGS_DIR / "subset.json"
    if default_subset.exists():
        data = json.loads(default_subset.read_text())
        return data["tasks"]

    print("No subset.json found. Run select_subset.py first.")
    sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-task evolve configs")
    parser.add_argument(
        "--subset",
        type=str,
        default=str(CONFIGS_DIR / "subset.json"),
        help="Path to subset.json (default: configs/subset.json)",
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_task_list(Path(args.subset))
    template_text = load_template()

    generated = 0
    for task_name in tasks:
        config = generate_config(task_name, template_text)
        output_path = CONFIGS_DIR / f"{task_name}.json"
        output_path.write_text(json.dumps(config, indent=2))
        generated += 1

    print(f"Generated {generated} config files in {CONFIGS_DIR}")
    for task in tasks:
        print(f"  - {task}.json")


if __name__ == "__main__":
    main()
