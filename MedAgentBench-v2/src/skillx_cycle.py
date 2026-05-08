"""
Entry point for the SkillX extraction-based skill comparator on MedAgentBench.
Extracts functional skills from successful trajectories after each epoch
and injects retrieved skills at inference.

Usage:
    python -m src.skillx_cycle --config configs/skillx_cycle.yaml --run-name run_001

The task worker must already be running before invoking this script.
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="SkillX cycle for MedAgentBench")
    parser.add_argument("--config", "-c", type=str, default="configs/skillx_cycle.yaml")
    parser.add_argument("--run-name", "-n", type=str, default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        print("--force and --resume are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs/skillx_cycle"))
    run_dir = output_dir / run_name

    if run_dir.exists() and not args.force and not args.resume:
        print(
            f"Run directory already exists: {run_dir}\n"
            "Use --resume to continue or --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    config["_resume"] = args.resume
    print(f"Run directory: {run_dir}")

    with (run_dir / "config.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cycle_cfg = config.get("cycle", {})
    skillx_cfg = config.get("skillx", {})
    print(f"Epochs:           {cycle_cfg.get('epochs')}")
    print(f"Val concurrency:  {cycle_cfg.get('batch_concurrency')} threads")
    print(f"Skill type:       {skillx_cfg.get('skill_type', 'functional')}")
    print(f"Retrieval top-k:  {skillx_cfg.get('retrieval_top_k', 5)}")
    print(f"Filter stage1:    {skillx_cfg.get('filter_stage1', True)}")
    print(f"Dev split:        {config['data']['dev']}")
    print(f"Val split:        {config['data']['val']}")

    from src.skillx.cycle import SkillXCycleRunner

    runner = SkillXCycleRunner(config=config, run_dir=run_dir)
    runner.run()


if __name__ == "__main__":
    main()
