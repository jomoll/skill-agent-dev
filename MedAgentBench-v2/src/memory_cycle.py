"""
Entry point for the sequential memory-learning cycle on MedAgentBench-v2.
Matches the original MedAgentBench-v2 paper: memory is updated after each failing sample.

Usage:
    python -m src.memory_cycle --config configs/memory_cycle.yaml --run-name run_001

The task worker must already be running before invoking this script.
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Memory-learning cycle for MedAgentBench-v2")
    parser.add_argument("--config", "-c", type=str, default="configs/memory_cycle.yaml")
    parser.add_argument("--run-name", "-n", type=str, default=None)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs/memory_cycle"))
    run_dir = output_dir / run_name

    if run_dir.exists() and not args.force:
        print(f"Run directory already exists: {run_dir}\nUse --force to overwrite.",
              file=sys.stderr)
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cycle_cfg = config.get("cycle", {})
    print(f"Epochs:          {cycle_cfg.get('epochs')}")
    print(f"Val concurrency: {cycle_cfg.get('batch_concurrency')} threads")
    print(f"Max bullets:     {config.get('memory', {}).get('max_bullets', 20)}")
    print(f"Dev split:       {config['data']['dev']}")
    print(f"Val split:       {config['data']['val']}")

    from src.memory.cycle import MemoryCycleRunner
    runner = MemoryCycleRunner(config=config, run_dir=run_dir)
    runner.run()


if __name__ == "__main__":
    main()
