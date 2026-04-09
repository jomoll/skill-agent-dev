"""
Entry point for the skill-learning cycle.

Usage:
    python -m src.skill_cycle --config configs/skill_cycle.yaml --run-name run_001

The task worker must already be running before invoking this script.
Start it with:
    python -m src.start_task --config configs/start_task.yaml -a
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Skill-learning cycle for MedAgentBench"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/skill_cycle.yaml",
        help="Path to skill_cycle config file",
    )
    parser.add_argument(
        "--run-name", "-n", type=str, default=None,
        help="Run name (default: timestamp)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Overwrite an existing run directory",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve run directory
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs/skill_cycle"))
    run_dir = output_dir / run_name

    if run_dir.exists() and not args.force:
        print(
            f"Run directory already exists: {run_dir}\n"
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Snapshot config into run dir
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Print reproducibility header
    cycle_cfg = config.get("cycle", {})
    print(f"Epochs:        {cycle_cfg.get('epochs')}")
    print(f"Update every:  {cycle_cfg.get('update_every')} samples")
    print(f"Concurrency:   {cycle_cfg.get('batch_concurrency')} threads")
    print(f"Max proposals: {cycle_cfg.get('max_proposals')}")
    print(f"Dev split:     {config['data']['dev']}")
    print(f"Val split:     {config['data']['val']}")

    # Launch runner
    from src.skills.cycle import SkillCycleRunner
    runner = SkillCycleRunner(config=config, run_dir=run_dir)
    runner.run()


if __name__ == "__main__":
    main()
