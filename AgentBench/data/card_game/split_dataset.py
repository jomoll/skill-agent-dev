"""
Generate dev/val split metadata for the Card Game skill cycle.

Card Game is procedurally generated — no external dataset. This script
produces split JSON files whose IDs match the indices from CardGame.get_indices()
when test_time=10 (40 total samples: 4 combos × 10 repetitions).

Split strategy: stratified 60/40 within each (baseline, agent_position) group
so both splits have balanced coverage of all game types.

  split_dev.json  — 24 samples (6 reps per combo group) — skill learning
  split_val.json  — 16 samples (4 reps per combo group) — monitoring

Run from AgentBench/:
  python data/card_game/split_dataset.py
"""

import json
from pathlib import Path

TEST_TIME = 10   # must match card-game-skill.parameters.test_time in card_game.yaml
DEV_PER_GROUP = 6  # 60% of 10 reps per combo
VAL_PER_GROUP = 4  # 40% of 10 reps per combo


def main():
    data_dir = Path("data/card_game")

    # Replicate CardGame.get_data() order to map indices to combos
    combos = [
        (2, "baseline1", 0),
        (2, "baseline1", 1),
        (2, "baseline2", 0),
        (2, "baseline2", 1),
    ]

    dev_data, val_data = [], []
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
            else:
                val_data.append(sample)
            idx += 1

    for split_name, data in [("dev", dev_data), ("val", val_data)]:
        out = data_dir / f"split_{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(data)} samples → {out}")

    print(f"\nSplit summary: {len(dev_data)} dev / {len(val_data)} val")
    print("(Total samples from card-game-skill with test_time=10: 40)")


if __name__ == "__main__":
    main()
