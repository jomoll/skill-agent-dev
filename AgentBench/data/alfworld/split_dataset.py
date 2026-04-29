"""
Split ALFWorld data into dev / val / test sets for the skill cycle.

Data sources:
  standard.json — 50 scenarios across 6 task types (dev + val pool)
  dev.json      — 20 additional held-out scenarios (test set)

Split strategy: stratified 60/40 within each task type from standard.json.

  split_dev.json   — 30 samples (6 task types × ~5 each) — skill learning
  split_val.json   — 20 samples (6 task types × ~3 each) — monitoring / early stopping
  split_test.json  — 20 samples from dev.json            — final evaluation

IDs:
  dev/val: sequential 0-based indices assigned in type-grouped order
  test: offset by len(standard) = 50 to avoid numeric collision

Usage:
    python data/alfworld/split_dataset.py
"""

import json
import random
from pathlib import Path

SEED = 42
DEV_RATIO = 0.6
DATA_DIR = Path(__file__).parent


def _scenario_to_sample(idx: int, task_type: str, path: str) -> dict:
    # Extract short description from path: the third path component is the scenario name
    # e.g. json_2.1.1/valid_unseen/pick_and_place_simple-Pencil-None-Shelf-308/trial_.../game.tw-pddl
    parts = path.split("/")
    description = parts[2] if len(parts) > 2 else path
    return {"id": idx, "description": description, "type": task_type}


def main():
    standard = json.load(open(DATA_DIR / "standard.json"))
    dev_raw = json.load(open(DATA_DIR / "dev.json"))

    rng = random.Random(SEED)

    dev_samples = []
    val_samples = []
    idx = 0

    for task_type, scenarios in sorted(standard.items()):
        shuffled = scenarios[:]
        rng.shuffle(shuffled)
        n_dev = max(1, int(len(shuffled) * DEV_RATIO))
        for path in shuffled[:n_dev]:
            dev_samples.append(_scenario_to_sample(idx, task_type, path))
            idx += 1
        for path in shuffled[n_dev:]:
            val_samples.append(_scenario_to_sample(idx, task_type, path))
            idx += 1
        print(f"  {task_type}: {n_dev} dev + {len(shuffled) - n_dev} val")

    # Test: dev.json scenarios, IDs offset by total standard count to avoid collision
    total_standard = sum(len(v) for v in standard.values())
    test_samples = []
    test_idx = total_standard
    for task_type, scenarios in sorted(dev_raw.items()):
        for path in scenarios:
            test_samples.append(_scenario_to_sample(test_idx, task_type, path))
            test_idx += 1

    print(f"\nDev:  {len(dev_samples)} samples")
    print(f"Val:  {len(val_samples)} samples")
    print(f"Test: {len(test_samples)} samples (from dev.json, held-out)")

    for split_name, data in [("dev", dev_samples), ("val", val_samples), ("test", test_samples)]:
        out = DATA_DIR / f"split_{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out}")

    # Verify no ID overlap
    dev_ids = {s["id"] for s in dev_samples}
    val_ids = {s["id"] for s in val_samples}
    test_ids = {s["id"] for s in test_samples}
    assert not (dev_ids & val_ids), "dev/val overlap"
    assert not (dev_ids & test_ids), "dev/test overlap"
    assert not (val_ids & test_ids), "val/test overlap"
    print("\nID overlap checks passed.")


if __name__ == "__main__":
    main()
