"""
Generate synthetic aggregation samples for DBBench dev split.

The four aggregation types (SUM, MIN, MAX, AVG) each have only 4 dev samples,
which is too few for reliable skill learning. This script generates additional
samples from the same tables already in standard.jsonl by varying the aggregated
column and WHERE conditions, verifying each answer with SQLite.

Synthetic samples are stored in synthetic_dev.json and included in split_dev.json
by split_dataset.py. IDs start from 10000 to avoid collision with real sample IDs.

Usage:
    python data/dbbench/generate_synthetic.py
"""

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

SEED = 42
TARGET_PER_TYPE = 20   # target dev samples per aggregation type after synthesis
DATA_DIR = Path(__file__).parent
SYNTHETIC_ID_OFFSET = 10000

AGG_TYPES = ["aggregation-SUM", "aggregation-MIN", "aggregation-MAX", "aggregation-AVG"]

NL_TEMPLATES = {
    "aggregation-SUM": [
        "What is the total {col} where {cond}?",
        "What is the sum of {col} for entries where {cond}?",
        "Give me the total {col} when {cond}.",
        "What is the total {col}?",
        "What is the sum of all {col} values?",
    ],
    "aggregation-MIN": [
        "What is the minimum {col} where {cond}?",
        "What is the smallest {col} when {cond}?",
        "What is the lowest {col} for entries where {cond}?",
        "What is the minimum {col}?",
        "What is the smallest {col} overall?",
    ],
    "aggregation-MAX": [
        "What is the maximum {col} where {cond}?",
        "What is the largest {col} when {cond}?",
        "What is the highest {col} for entries where {cond}?",
        "What is the maximum {col}?",
        "What is the largest {col} overall?",
    ],
    "aggregation-AVG": [
        "What is the average {col} where {cond}?",
        "What is the mean {col} when {cond}?",
        "What is the average {col} for entries where {cond}?",
        "What is the average {col}?",
        "What is the mean {col} overall?",
    ],
}

SQL_FUNC = {
    "aggregation-SUM": "SUM",
    "aggregation-MIN": "MIN",
    "aggregation-MAX": "MAX",
    "aggregation-AVG": "AVG",
}


def _load_table(conn, columns, rows):
    col_defs = ", ".join(f'"{c["name"]}" TEXT' for c in columns)
    conn.execute(f"CREATE TABLE t ({col_defs})")
    placeholders = ", ".join("?" for _ in columns)
    for row in rows:
        try:
            conn.execute(f"INSERT INTO t VALUES ({placeholders})", [str(v) if v is not None else "" for v in row])
        except Exception:
            pass


def _run_agg(conn, func, col, cond_sql=None):
    sql = f'SELECT {func}(CAST("{col}" AS REAL)) FROM t'
    if cond_sql:
        sql += f" WHERE {cond_sql}"
    try:
        val = conn.execute(sql).fetchone()[0]
        if val is None:
            return None
        f = float(val)
        # Reject zero results and results that look like a year (likely a non-numeric column)
        if f == 0.0:
            return None
        return str(round(f, 1)) if f != int(f) else str(float(int(f)))
    except Exception:
        return None


def _get_column_values(conn, col, limit=10):
    try:
        rows = conn.execute(f'SELECT DISTINCT "{col}" FROM t WHERE "{col}" IS NOT NULL AND "{col}" != "" LIMIT {limit}').fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def _is_numeric_column(conn, col):
    vals = _get_column_values(conn, col, limit=20)
    if not vals:
        return False
    numeric = 0
    for v in vals:
        try:
            float(str(v).replace(",", ""))
            numeric += 1
        except ValueError:
            pass
    return numeric / len(vals) >= 0.7


def generate_from_sample(sample, agg_type, rng, existing_descs):
    ti = sample["table"]["table_info"]
    columns = ti["columns"]
    rows = ti["rows"]
    col_names = [c["name"] for c in columns]

    conn = sqlite3.connect(":memory:")
    _load_table(conn, columns, rows)

    func = SQL_FUNC[agg_type]
    templates = NL_TEMPLATES[agg_type]
    generated = []

    # Identify numeric columns
    numeric_cols = [c for c in col_names if _is_numeric_column(conn, c)]
    if not numeric_cols:
        conn.close()
        return generated

    # Text columns suitable for WHERE conditions
    text_cols = [c for c in col_names if c not in numeric_cols]

    for num_col in numeric_cols:
        # No-condition variant
        answer = _run_agg(conn, func, num_col)
        if answer:
            tmpl = rng.choice(templates[-2:])  # unconditional templates
            desc = tmpl.format(col=num_col)
            if desc not in existing_descs:
                existing_descs.add(desc)
                generated.append({"description": desc, "type": agg_type, "answer": answer})

        # One-condition variants
        for filter_col in rng.sample(text_cols, min(3, len(text_cols))):
            vals = _get_column_values(conn, filter_col, limit=8)
            if not vals:
                continue
            for filter_val in rng.sample(vals, min(3, len(vals))):
                safe_val = filter_val.replace("'", "''")
                cond_sql = f'"{filter_col}" = \'{safe_val}\''
                answer = _run_agg(conn, func, num_col, cond_sql)
                if answer:
                    cond_nl = f"{filter_col} is {filter_val}"
                    tmpl = rng.choice(templates[:3])  # conditional templates
                    desc = tmpl.format(col=num_col, cond=cond_nl)
                    if desc not in existing_descs:
                        existing_descs.add(desc)
                        generated.append({"description": desc, "type": agg_type, "answer": answer})

    conn.close()
    return generated


def main():
    rng = random.Random(SEED)

    with open(DATA_DIR / "standard.jsonl") as f:
        standard = [json.loads(l) for l in f if l.strip()]

    # Count existing dev samples per type
    with open(DATA_DIR / "split_dev.json") as f:
        dev = json.load(f)
    existing_per_type = defaultdict(int)
    existing_descs = set(s["description"] for s in dev)
    for s in dev:
        existing_per_type[s["type"]] += 1

    print("Existing dev counts per aggregation type:")
    for t in AGG_TYPES:
        print(f"  {t}: {existing_per_type[t]}")

    # Collect all aggregation samples from standard.jsonl
    by_type = defaultdict(list)
    for s in standard:
        t = s["type"][0]
        if t in AGG_TYPES:
            by_type[t].append(s)

    synthetic = []
    next_id = SYNTHETIC_ID_OFFSET

    for agg_type in AGG_TYPES:
        needed = max(0, TARGET_PER_TYPE - existing_per_type[agg_type])
        print(f"\n{agg_type}: need {needed} more samples")
        candidates = []
        # Try all available tables for this type
        pool = by_type[agg_type][:]
        rng.shuffle(pool)
        for sample in pool:
            new = generate_from_sample(sample, agg_type, rng, existing_descs)
            candidates.extend(new)
            if len(candidates) >= needed * 3:
                break

        rng.shuffle(candidates)
        selected = candidates[:needed]
        for s in selected:
            s["id"] = next_id
            next_id += 1
        synthetic.extend(selected)
        print(f"  generated {len(selected)} samples")

    with open(DATA_DIR / "synthetic_dev.json", "w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(synthetic)} synthetic samples → {DATA_DIR / 'synthetic_dev.json'}")

    # Summary
    by_type_out = defaultdict(int)
    for s in synthetic:
        by_type_out[s["type"]] += 1
    for t in AGG_TYPES:
        total = existing_per_type[t] + by_type_out[t]
        print(f"  {t}: {existing_per_type[t]} real + {by_type_out[t]} synthetic = {total} total dev")


if __name__ == "__main__":
    main()
