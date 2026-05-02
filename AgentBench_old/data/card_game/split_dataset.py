"""
Generate dev/val/test split metadata for the Card Game skill cycle.

Card Game is procedurally generated — no external dataset. This script
produces split JSON files whose IDs match the indices from CardGame.get_indices()
when test_time=25 (100 total samples: 4 combos × 25 repetitions).

Split strategy: within each (baseline, agent_position) combo group:
  reps  0-14 → dev  (15 reps × 4 combos = 60 samples) — skill learning
  reps 15-19 → val  ( 5 reps × 4 combos = 20 samples) — monitoring / early stopping
  reps 20-24 → test ( 5 reps × 4 combos = 20 samples) — final evaluation

Run from AgentBench/:
  python data/card_game/split_dataset.py
"""

import json
from pathlib import Path

TEST_TIME = 25       # must match card-game-skill.parameters.test_time in card_game.yaml
DEV_PER_GROUP = 15   # reps 0-14
VAL_PER_GROUP = 5    # reps 15-19
TEST_PER_GROUP = 5   # reps 20-24


def main():
    data_dir = Path("data/card_game")

    # Replicate CardGame.get_data() order to map indices to combos
    combos = [
        (2, "baseline1", 0),
        (2, "baseline1", 1),
        (2, "baseline2", 0),
        (2, "baseline2", 1),
    ]

    dev_data, val_data, test_data = [], [], []
    idx = 0
    for stage, base, agent in combos:
        for rep in range(TEST_TIME):
            sample = {
                "id": idx,
                "description": f"Card Game stage={stage} vs {base}, agent_position={agent}, rep={rep}",
                "stage": stage,
                "baseline": base,
                "agent": agent,
            }
            if rep < DEV_PER_GROUP:
                dev_data.append(sample)
            elif rep < DEV_PER_GROUP + VAL_PER_GROUP:
                val_data.append(sample)
            else:
                test_data.append(sample)
            idx += 1

    for split_name, data in [("dev", dev_data), ("val", val_data), ("test", test_data)]:
        out = data_dir / f"split_{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(data)} samples → {out}")

    print(f"\nSplit summary: {len(dev_data)} dev / {len(val_data)} val / {len(test_data)} test")
    print(f"(Total samples from card-game-skill with test_time={TEST_TIME}: {TEST_TIME * len(combos)})")


if __name__ == "__main__":
    main()
