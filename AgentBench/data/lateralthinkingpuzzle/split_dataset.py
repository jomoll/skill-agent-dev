"""
Split the LTP dataset into dev/val/test splits for the skill cycle.

Splits:
  split_dev.json   — 30 samples (60% of standard.xlsx, stratified by random seed)
  split_val.json   — 20 samples (40% of standard.xlsx)
  split_test.json  — 20 samples (dev.xlsx, held out; evaluated via ltp-dev task)

IDs in split_dev.json and split_val.json correspond to row indices in standard.xlsx
(0-based), which is what ltp-skill's start_sample(index) expects.

IDs in split_test.json correspond to row indices in dev.xlsx (0-based), which
is what ltp-dev's start_sample(index) expects.

Run from AgentBench/:
  python data/lateralthinkingpuzzle/split_dataset.py
"""

import json
import random
from pathlib import Path

import openpyxl


def read_xlsx(path: Path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    # First row is header: story, answer, Story keys, Answer keys
    return [
        {
            "id": i,
            "story": str(row[0]) if row[0] is not None else "",
            "answer": str(row[1]) if row[1] is not None else "",
        }
        for i, row in enumerate(rows[1:])
    ]


def main():
    data_dir = Path("data/lateralthinkingpuzzle")

    # Load standard.xlsx (50 rows) — training / eval set
    standard = read_xlsx(data_dir / "standard.xlsx")
    print(f"Loaded {len(standard)} samples from standard.xlsx")

    # Load dev.xlsx (20 rows) — held-out test set
    test_data = read_xlsx(data_dir / "dev.xlsx")
    print(f"Loaded {len(test_data)} samples from dev.xlsx (test)")

    # 60/40 split of standard.xlsx
    rng = random.Random(42)
    indices = list(range(len(standard)))
    rng.shuffle(indices)

    split_n = int(len(standard) * 0.6)  # 30
    dev_indices = sorted(indices[:split_n])
    val_indices = sorted(indices[split_n:])

    dev_data = [standard[i] for i in dev_indices]
    val_data = [standard[i] for i in val_indices]

    for split_name, data in [("dev", dev_data), ("val", val_data), ("test", test_data)]:
        out = data_dir / f"split_{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(data)} samples → {out}")

    print(
        f"\nSplit summary: {len(dev_data)} dev / {len(val_data)} val / {len(test_data)} test"
    )


if __name__ == "__main__":
    main()
