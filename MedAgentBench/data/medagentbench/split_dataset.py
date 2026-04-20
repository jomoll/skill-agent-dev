"""
Dataset split for MedAgentBench (test_data_v2.json)

Test set  : tasks 6, 7, 10  (all 30 samples each, held-out)
Dev set   : 60% of each remaining task (tasks 1–5, 8, 9), shuffled within task
Val set   : remaining 40% of each remaining task

Fixed seed for reproducibility: SEED = 42
"""

import json
import random
from collections import defaultdict
from pathlib import Path
import tempfile

SEED = 42
DEV_RATIO = 0.6
TEST_TASKS = {"task6", "task7", "task10"}

DATA_DIR = Path(__file__).parent
INPUT_FILE = DATA_DIR / "test_data_v2.json"
OUTPUT_FILES = {
    "test": DATA_DIR / "split_test.json",
    "dev":  DATA_DIR / "split_dev.json",
    "val":  DATA_DIR / "split_val.json",
}


def get_task_id(sample: dict) -> str:
    return sample["id"].rsplit("_", 1)[0]


def _atomic_dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.stem}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Group by task
    by_task = defaultdict(list)
    for sample in data:
        by_task[get_task_id(sample)].append(sample)

    test, dev, val = [], [], []

    rng = random.Random(SEED)

    for task, samples in sorted(by_task.items()):
        if task in TEST_TASKS:
            test.extend(samples)
        else:
            shuffled = samples[:]
            rng.shuffle(shuffled)
            n_dev = int(len(shuffled) * DEV_RATIO)
            dev.extend(shuffled[:n_dev])
            val.extend(shuffled[n_dev:])

    # Assertions
    assert len(test) + len(dev) + len(val) == len(data), "Split sizes do not sum to total"
    assert set(get_task_id(s) for s in test) == TEST_TASKS, "Test set contains unexpected tasks"
    assert not (set(get_task_id(s) for s in dev) & TEST_TASKS), "Dev set contains held-out tasks"
    assert not (set(get_task_id(s) for s in val) & TEST_TASKS), "Val set contains held-out tasks"
    assert not {s["id"] for s in dev} & {s["id"] for s in val}, "Dev and val sets overlap"

    for split_name, split_data in [("test", test), ("dev", dev), ("val", val)]:
        _atomic_dump_json(OUTPUT_FILES[split_name], split_data)

    # Reproducibility log
    print(f"Input:      {INPUT_FILE}")
    print(f"Seed:       {SEED}")
    print(f"Dev ratio:  {DEV_RATIO}")
    print(f"Test tasks: {sorted(TEST_TASKS)}")
    print()

    # Summary
    dev_by_task  = defaultdict(int)
    val_by_task  = defaultdict(int)
    test_by_task = defaultdict(int)
    for s in dev:  dev_by_task[get_task_id(s)]  += 1
    for s in val:  val_by_task[get_task_id(s)]  += 1
    for s in test: test_by_task[get_task_id(s)] += 1

    all_tasks = sorted(by_task.keys(), key=lambda x: int(x.replace("task", "")))
    print(f"{'task':<10} {'dev':>6} {'val':>6} {'test':>6}")
    print("-" * 30)
    for task in all_tasks:
        print(f"{task:<10} {dev_by_task[task]:>6} {val_by_task[task]:>6} {test_by_task[task]:>6}")
    print("-" * 30)
    print(f"{'TOTAL':<10} {len(dev):>6} {len(val):>6} {len(test):>6}")
    print(f"\nWrote: {[str(OUTPUT_FILES[s]) for s in ('dev','val','test')]}")


if __name__ == "__main__":
    main()
