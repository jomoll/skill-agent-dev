import argparse
import datetime
import io
import json
import random
import shutil
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from core_utils import curate_input_dataset, parse_outputs

from .agent import (
    LiteLLMAgent,
    create_skill_aware_fhir_agent,
    format_agent_actions,
    serialize_message,
)
from .evaluator import FHIRSampleEvaluator
from .repository import SkillRepository
from .updater import SkillUpdater


class _TeeStream(io.TextIOBase):
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, s):
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")


def _load_samples(csv_path: Path, split_names: Iterable[str], limit: Optional[int]) -> List[Dict]:
    df = pd.read_csv(csv_path)
    df = df[df["split"].isin(list(split_names))].copy()
    if limit:
        df = df.head(limit)
    df["question_with_context"] = curate_input_dataset(df, add_patient_fhir_id=True)
    return df.to_dict("records")


def _compute_skill_effectiveness(
    entries: List[Dict], prev_results: Optional[Dict[str, bool]]
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for entry in entries:
        sid = str(entry.get("sample_id"))
        now = bool(entry.get("is_correct"))
        prev = prev_results.get(sid) if prev_results else None
        for skill in entry.get("skill_snapshot_before") or []:
            name = skill.get("name")
            if not name or name == "skeleton":
                continue
            stats.setdefault(name, {"fixes": 0, "regressions": 0, "runs": 0})
            stats[name]["runs"] += 1
            if prev is not None:
                if not prev and now:
                    stats[name]["fixes"] += 1
                elif prev and not now:
                    stats[name]["regressions"] += 1
    return stats


class FHIRSkillCycleRunner:
    def __init__(self, config: Dict, run_dir: Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.output_dir = self.run_dir

        agent_cfg = config["agent"]
        self.agent_strategy = agent_cfg.get("strategy", "multi_turn_resource")
        self.agent_model = agent_cfg["model"]
        self.agent_base_url = agent_cfg.get("base_url")
        self.verbose_agent = bool(agent_cfg.get("verbose", False))

        updater_cfg = config.get("updater", {})
        self.updater_agent = LiteLLMAgent(
            model=updater_cfg.get("model", self.agent_model),
            base_url=updater_cfg.get("base_url", self.agent_base_url),
            temperature=float(updater_cfg.get("temperature", 0.0)),
            max_tokens=int(updater_cfg.get("max_tokens", 32000)),
        )

        eval_cfg = config.get("eval", {})
        self.evaluator = FHIRSampleEvaluator(
            model=eval_cfg.get("model", self.agent_model),
            base_url=eval_cfg.get("base_url", self.agent_base_url),
            cache_path=self.run_dir / "eval_cache.json",
        )

        cycle_cfg = config["cycle"]
        self.epochs = int(cycle_cfg.get("epochs", 3))
        self.update_every = int(cycle_cfg.get("update_every", 25))
        self.batch_concurrency = int(cycle_cfg.get("batch_concurrency", 4))
        self.grpo_k = int(cycle_cfg.get("grpo_k", 4))
        self.grpo_eval_n = int(cycle_cfg.get("grpo_eval_n", 20))
        self.run_baseline = bool(cycle_cfg.get("run_baseline", True))
        self.max_proposals = int(cycle_cfg.get("max_proposals", 1))
        self.max_learned_skills = int(cycle_cfg.get("max_learned_skills", 10))

        data_cfg = config["data"]
        csv_path = Path(data_cfg["csv"])
        self.dev_data = _load_samples(csv_path, data_cfg.get("dev_splits", ["train"]), data_cfg.get("dev_limit"))
        self.val_data = _load_samples(csv_path, data_cfg.get("val_splits", ["valid"]), data_cfg.get("val_limit"))

        skills_cfg = config["skills"]
        self.skill_repo = SkillRepository(
            base_dir=Path(skills_cfg["base_dir"]),
            learned_dir=self.run_dir / "skills" / "learned",
        )
        self.updater = SkillUpdater(
            self.updater_agent,
            max_proposals=self.max_proposals,
            max_learned_skills=self.max_learned_skills,
        )
        self._progress_stream = None

    def run(self) -> None:
        log_path = self.run_dir / "run.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        self._progress_stream = (
            original_stderr
            if tqdm is not None and getattr(original_stderr, "isatty", lambda: False)()
            else None
        )
        
        # NOTE: Hide background scores, skill proposals, and regression scores from CLI
        # by defaulting sys.stdout to log_file.
        # But we will temporarily swap it back during _update_skills trace generation!
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            self._run_inner()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self._progress_stream = None
            log_file.close()

    def _progress(self, iterable, *, total: Optional[int] = None, desc: str = "", leave: bool = False):
        if tqdm is None or self._progress_stream is None:
            return iterable
        return tqdm(iterable, total=total, desc=desc, leave=leave, file=self._progress_stream, dynamic_ncols=True)

    def _run_inner(self) -> None:
        print(f"[FHIRSkillCycle] dev={len(self.dev_data)} val={len(self.val_data)} strategy={self.agent_strategy}")
        
        val_scores = []
        if self.run_baseline:
            baseline_dir = self.run_dir / "baseline"
            baseline_dir.mkdir(exist_ok=True)
            score = self._evaluate_split(self.val_data, baseline_dir / "val_runs.jsonl", update_cycle=-1)
            print(f"[Baseline] Val: {score:.1%}")
            val_scores.append({"epoch": -1, "score": score})
            (self.run_dir / "val_scores.json").write_text(json.dumps(val_scores, indent=2), encoding="utf-8")

        prev_taxonomy: Dict[str, str] = {}
        prev_results: Optional[Dict[str, bool]] = None
        for epoch in range(self.epochs):
            print(f"\n{'=' * 60}\n  EPOCH {epoch}\n{'=' * 60}")
            epoch_dir = self.run_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            entries, prev_taxonomy = self._run_epoch(epoch, epoch_dir, prev_results, prev_taxonomy)
            prev_results = {str(e["sample_id"]): bool(e["is_correct"]) for e in entries}
            val_score = self._evaluate_split(self.val_data, epoch_dir / "val_runs.jsonl", update_cycle=epoch)
            val_scores.append({"epoch": epoch, "score": val_score})
            (epoch_dir / "val_score.json").write_text(
                json.dumps({"epoch": epoch, "score": val_score}, indent=2),
                encoding="utf-8",
            )
            (self.run_dir / "val_scores.json").write_text(json.dumps(val_scores, indent=2), encoding="utf-8")
            print(f"[Epoch {epoch}] Val: {val_score:.1%}")

    def _run_epoch(
        self,
        epoch: int,
        epoch_dir: Path,
        prev_results: Optional[Dict[str, bool]],
        prev_taxonomy: Dict[str, str],
    ) -> Tuple[List[Dict], Dict[str, str]]:
        rng = random.Random(epoch)
        dev = self.dev_data[:]
        rng.shuffle(dev)
        batches = [dev[i:i + self.update_every] for i in range(0, len(dev), self.update_every)]
        print(f"[Epoch {epoch}] {len(dev)} dev samples, {len(batches)} batches")

        all_entries: List[Dict] = []
        updates: List[Dict] = []
        dev_runs_path = epoch_dir / "dev_runs.jsonl"
        for batch_id, batch in enumerate(self._progress(batches, total=len(batches), desc=f"Epoch {epoch} batches")):
            print(f"  Batch {batch_id}/{len(batches)-1}: {len(batch)} samples")
            batch_entries = self._run_samples(batch, self.skill_repo, update_cycle=batch_id)
            self._append_jsonl(dev_runs_path, batch_entries)
            all_entries.extend(batch_entries)
            print(f"  Batch score: {sum(e['is_correct'] for e in batch_entries)}/{len(batch_entries)}")

            event = self._update_skills(
                batch_entries=batch_entries,
                all_entries=all_entries,
                prev_results=prev_results,
                prev_taxonomy=prev_taxonomy,
                epoch=epoch,
                update_cycle=batch_id,
            )
            if event:
                updates.append(event)
                prev_taxonomy.update(event.get("new_failure_labels", {}))
            (epoch_dir / "skill_updates.json").write_text(json.dumps(updates, indent=2), encoding="utf-8")

        return all_entries, prev_taxonomy

    def _update_skills(
        self,
        *,
        batch_entries: List[Dict],
        all_entries: List[Dict],
        prev_results: Optional[Dict[str, bool]],
        prev_taxonomy: Dict[str, str],
        epoch: int,
        update_cycle: int,
    ) -> Optional[Dict]:
        failing = [e for e in batch_entries if not e.get("is_correct")]
        if not failing:
            print("  No failures in batch; skipping update.")
            return None

        sample_to_label, new_labels = self.updater.classify_failures(failing, prev_taxonomy)
        diagnosis = self.updater.diagnose(failing, self.skill_repo, failure_labels=sample_to_label)
        effectiveness = _compute_skill_effectiveness(all_entries, prev_results)
        groups: Dict[str, List[Dict]] = {}
        for entry in failing:
            groups.setdefault(sample_to_label.get(str(entry["sample_id"]), "unclassified_failure"), []).append(entry)

        applied_all = []
        event = {"epoch": epoch, "update_cycle": update_cycle, "groups": [], "new_failure_labels": new_labels}
        for label, group_entries in groups.items():
            proposals = []
            for _ in range(self.grpo_k):
                raw = self.updater.propose(
                    group_entries,
                    self.skill_repo,
                    prev_results=prev_results,
                    skill_effectiveness=effectiveness,
                    failure_mode=label,
                    diagnosis=diagnosis,
                )
                valid = self.updater.validate(raw, self.skill_repo)
                if valid:
                    proposals.append(valid)

            if not proposals:
                event["groups"].append({"label": label, "applied": [], "reason": "no_valid_proposals"})
                continue

            probe = self._build_probe(group_entries, all_entries)
            baseline_score = sum(e.get("is_correct", False) for e in probe)
            best = None
            best_score = baseline_score
            for candidate in proposals:
                fork = self.skill_repo.fork()
                try:
                    self.updater.apply(candidate, fork)
                    probe_samples = [e["_sample"] for e in probe]
                    print(
                        f"  [ProposalRanking] evaluating {len(probe_samples)} "
                        f"probe samples for {', '.join(p['name'] for p in candidate)}"
                    )
                    probe_entries = self._run_samples(probe_samples, fork, update_cycle=update_cycle)
                    score = sum(e.get("is_correct", False) for e in probe_entries)
                except Exception as e:
                    print(f"  [ProposalRanking] candidate failed: {e}")
                    event["groups"].append({
                        "label": label,
                        "candidate": [
                            {k: v for k, v in p.items() if not k.startswith("_")}
                            for p in candidate
                        ],
                        "error": str(e),
                    })
                    score = -1
                finally:
                    fork.cleanup()
                if score > best_score:
                    best = candidate
                    best_score = score

            if best is None:
                event["groups"].append({
                    "label": label,
                    "baseline_score": baseline_score,
                    "best_score": best_score,
                    "applied": [],
                })
                continue

            for proposal in best:
                proposal["_provenance"] = {
                    "epoch": epoch,
                    "update_cycle": update_cycle,
                    "failure_mode": label,
                    "probe_score": best_score - baseline_score,
                    "fixes": best_score,
                    "regressions": max(0, baseline_score - best_score),
                }
            applied = self.updater.apply(best, self.skill_repo)
            applied_all.extend(applied)
            event["groups"].append({
                "label": label,
                "baseline_score": baseline_score,
                "best_score": best_score,
                "applied": applied,
            })
            print(f"  [SkillUpdate] {label}: applied {len(applied)} edit(s), probe {baseline_score}->{best_score}")

        event["applied"] = applied_all
        return event

    def _build_probe(self, group_entries: List[Dict], all_entries: List[Dict]) -> List[Dict]:
        selected = list(group_entries)
        passing = [e for e in all_entries if e.get("is_correct")]
        failing_other = [e for e in all_entries if not e.get("is_correct") and e not in selected]
        random.Random(0).shuffle(passing)
        random.Random(1).shuffle(failing_other)
        target = max(len(selected), min(self.grpo_eval_n, len(all_entries)))
        for pool in (passing, failing_other):
            for entry in pool:
                if len(selected) >= target:
                    break
                selected.append(entry)
        return selected[:target]

    def _evaluate_split(self, samples: List[Dict], path: Path, update_cycle: int) -> float:
        print(f"[Eval] evaluating {len(samples)} samples -> {path}")
        entries = self._run_samples(
            samples,
            self.skill_repo,
            update_cycle=update_cycle,
            append_path=path,
        )
        if not entries:
            return 0.0
        return sum(e["is_correct"] for e in entries) / len(entries)

    def _run_samples(
        self,
        samples: List[Dict],
        repo: SkillRepository,
        update_cycle: int,
        append_path: Optional[Path] = None,
    ) -> List[Dict]:
        results: List[Dict] = []
        if append_path:
            append_path.parent.mkdir(parents=True, exist_ok=True)
            append_path.write_text("", encoding="utf-8")
        with ThreadPoolExecutor(max_workers=self.batch_concurrency) as executor:
            futures = {
                executor.submit(self._run_one, sample, repo, update_cycle): sample
                for sample in samples
            }
            for future in self._progress(as_completed(futures), total=len(futures), desc="FHIR samples"):
                sample = futures[future]
                try:
                    entry = future.result()
                except Exception as e:
                    print(
                        f"[FHIRSkillCycle] sample runner failed for "
                        f"{sample.get('question_id')}: {e}"
                    )
                    entry = {
                        "sample_id": sample.get("question_id"),
                        "instruction": sample.get("question"),
                        "query_type": sample.get("template") or sample.get("main_table_name"),
                        "is_correct": False,
                        "update_cycle": update_cycle,
                        "status": "runner_error",
                        "error": str(e),
                        "ground_truth": sample.get("true_answer"),
                        "task_result": {},
                        "agent_actions": [],
                        "history": [],
                        "failure_tags": [
                            str(x)
                            for x in [sample.get("template"), sample.get("main_table_name")]
                            if x
                        ],
                        "skill_snapshot_before": repo.snapshot(),
                        "_sample": sample,
                    }
                results.append(entry)
                if append_path:
                    with open(append_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(self._json_safe(entry), default=str) + "\n")
                    print(
                        f"[RunSamples] {len(results)}/{len(samples)} complete "
                        f"score={sum(bool(e.get('is_correct')) for e in results)}/{len(results)}"
                    )
        results.sort(key=lambda e: str(e["sample_id"]))
        return results

    def _run_one(self, sample: Dict, repo: SkillRepository, update_cycle: int) -> Dict:
        import tools.cache as cache_module

        cache_module.CACHE_ENABLED = bool(self.config.get("agent", {}).get("enable_cache", True))
        agent = create_skill_aware_fhir_agent(
            agent_strategy=self.agent_strategy,
            model=self.agent_model,
            base_url=self.agent_base_url,
            verbose=self.verbose_agent,
            skill_repo=repo,
        )
        try:
            raw_output = agent.run(sample["question_with_context"])
            parsed = parse_outputs(raw_output)
        except Exception as e:
            raw_output = {"error": str(e), "trace": []}
            parsed = {"agent_answer": None, "agent_fhir_resources": None, "trace": [], "usage": None, "error": str(e)}

        trace = [serialize_message(m) for m in (parsed.get("trace") or raw_output.get("trace") or []) if m is not None]
        is_correct = self.evaluator.score(sample, parsed)
        return {
            "sample_id": sample["question_id"],
            "instruction": sample["question"],
            "query_type": sample.get("template") or sample.get("main_table_name"),
            "is_correct": is_correct,
            "update_cycle": update_cycle,
            "status": "completed" if not parsed.get("error") else "agent_error",
            "error": parsed.get("error"),
            "ground_truth": sample.get("true_answer"),
            "task_result": {
                "reported_answer": parsed.get("agent_answer"),
                "retrieved_fhir_resources": parsed.get("agent_fhir_resources"),
                "usage": parsed.get("usage"),
            },
            "agent_actions": format_agent_actions(trace),
            "history": trace,
            "failure_tags": [str(x) for x in [sample.get("template"), sample.get("main_table_name")] if x],
            "skill_snapshot_before": repo.snapshot(),
            "_sample": sample,
        }

    @staticmethod
    def _json_safe(entry: Dict) -> Dict:
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def _write_jsonl(self, path: Path, entries: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(self._json_safe(entry), default=str) + "\n")

    def _append_jsonl(self, path: Path, entries: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(self._json_safe(entry), default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-learning cycle for FHIR-AgentBench")
    parser.add_argument("--config", "-c", default="configs/skill_cycle.yaml")
    parser.add_argument("--run-name", "-n", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("output_dir", "outputs/skill_cycle")) / run_name
    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)
    elif run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}. Use --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")
    print(f"Run directory: {run_dir}")

    FHIRSkillCycleRunner(config, run_dir).run()


if __name__ == "__main__":
    main()
