"""
SkillCycleRunner — the main training loop for autonomous skill learning on
AgentBench tasks (OS Interaction, DBBench, …).

The eval function is pluggable via the config's `eval.module` key, which must
point to a Python module exposing `eval(sample, task_output) -> bool`.

Loop structure (per epoch):
    1. Shuffle dev samples (seed = epoch index for reproducibility)
    2. Split into batches of size `update_every`
    3. For each batch:
        a. Run all samples in parallel (up to `batch_concurrency` threads)
        b. Log results with update_cycle annotation
        c. Grouped proposal ranking / best-of-K skill update:
           - Sample K single-change proposals from the updater (temperature > 0)
           - Build a balanced eval set: up to `grpo_eval_n/2` failing + passing samples
           - For each proposal: fork the repo, apply the change, run the eval set
           - Apply the proposal with the best weighted score on the probe set
           - If no proposal improves the score, apply nothing
    4. Evaluate silently on the val set (no skill updates from val)
    5. Write per-epoch summary

Output layout (inside run_dir/):
    config.yaml
    skills/learned/
    epoch_0/
        dev_runs.jsonl        one JSON line per completed sample
        skill_updates.json    list of update events {cycle, proposals, applied, grpo}
        val_score.json        {epoch, score, n_correct, n_total}
    epoch_1/
        ...
    val_scores.json           [{epoch, score}] learning curve
"""

import importlib
import io
import json
import random
from copy import deepcopy
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.client.agents.skill_aware_agent import SkillAwareAgent
from src.client.task import TaskClient
from src.skills.repository import SkillRepository
from src.skills.updater import SkillUpdater
from src.typings import TaskClientOutput


_INVALID_ACTION_STATUS = "agent invalid action"
_INVALID_ACTION_REGRESSION_PENALTY = 2


def _load_eval_fn(config: Dict) -> Optional[Callable]:
    """Load the task-specific eval function from config['eval']['module']."""
    eval_module = config.get("eval", {}).get("module")
    if not eval_module:
        print("[SkillCycle] Warning: no eval.module in config — scores will always be False")
        return None
    try:
        mod = importlib.import_module(eval_module)
        return getattr(mod, "eval")
    except Exception as e:
        print(f"[SkillCycle] Warning: could not import eval from {eval_module}: {e}")
        return None


def _score_result(sample: Dict, result: TaskClientOutput,
                  eval_fn: Optional[Callable]) -> bool:
    """Return True if the task was solved correctly."""
    if result.error or result.output is None:
        return False
    if result.output.result is None:
        return False
    if eval_fn is None:
        return False
    try:
        direct = eval_fn(sample, result.output) is True
        if direct:
            return True

        fallback_output = _build_history_answer_fallback(sample, result.output)
        if fallback_output is not None:
            return eval_fn(sample, fallback_output) is True
        return False
    except Exception as e:
        print(f"[SkillCycle] eval error for {sample.get('id')}: {e}")
        return False


def _load_required_json_list(path: Path, label: str) -> List[Dict]:
    if not path.exists():
        raise ValueError(f"[SkillCycle] required {label} file does not exist: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"[SkillCycle] failed to read {label} at {path}: {e}") from e
    if not raw.strip():
        raise ValueError(
            f"[SkillCycle] required {label} file is empty: {path}. "
            "Regenerate the dataset split before running skill_cycle."
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[SkillCycle] required {label} file is invalid JSON at {path}: {e}"
        ) from e
    if not isinstance(data, list):
        raise ValueError(
            f"[SkillCycle] required {label} file must contain a JSON array, "
            f"got {type(data).__name__}: {path}"
        )
    return data


def _load_json_list_or_empty(path: Path, label: str) -> List[Dict]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"[SkillCycle] warning: failed to read {label} at {path}: {e}")
        return []
    if not raw:
        print(f"[SkillCycle] warning: {label} at {path} is empty; treating as []")
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(
            f"[SkillCycle] warning: {label} at {path} is invalid JSON "
            f"({e}); treating as []"
        )
        return []
    if not isinstance(data, list):
        print(
            f"[SkillCycle] warning: {label} at {path} is a "
            f"{type(data).__name__}, expected list; treating as []"
        )
        return []
    return data


def _extract_final_answer_text(agent_actions: List[str]) -> str:
    for action in reversed(agent_actions or []):
        if "Final Answer:" in action:
            return action.split("Final Answer:", 1)[1].strip()
    return ""


def _extract_final_answer_from_history(history: List[Dict]) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        role = msg.role if hasattr(msg, "role") else msg.get("role")
        if role != "agent":
            continue
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if "Final Answer:" in str(content):
            return str(content).split("Final Answer:", 1)[1].strip()
    return ""


def _is_dbbench_like(sample: Dict, task_output) -> bool:
    if "type" not in sample:
        return False
    history = getattr(task_output, "history", None) or []
    for msg in history[:3]:
        role = msg.role if hasattr(msg, "role") else msg.get("role")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if role == "user" and "help me operate a mysql database with sql" in str(content).lower():
            return True
    return False


def _build_history_answer_fallback(sample: Dict, task_output):
    if not _is_dbbench_like(sample, task_output):
        return None
    history = getattr(task_output, "history", None) or []
    final_answer = _extract_final_answer_from_history(history)
    if not final_answer:
        return None
    cloned = deepcopy(task_output)
    cloned.result = dict(getattr(task_output, "result", {}) or {})
    cloned.result["answer"] = final_answer
    cloned.result["answer_source"] = "history_final_answer"
    return cloned


def _infer_failure_tags(sample: Dict, result: TaskClientOutput, agent_actions: List[str]) -> List[str]:
    tags: List[str] = []
    if result.error:
        tags.append("task_error")

    output = result.output
    status = output.status if output else "error"
    query_type = str(sample.get("type", "other"))
    is_mutation = query_type in ("INSERT", "UPDATE", "DELETE")
    action_text = "\n".join(agent_actions or [])
    final_answer = _extract_final_answer_text(agent_actions)
    final_lower = final_answer.lower()

    if status == "agent validation failed":
        tags.append("protocol_invalid")
        if "Action: Operation" not in action_text and "Action: Answer" not in action_text:
            tags.append("no_valid_action")
        elif "Action: Operation" in action_text and "Action: Answer" not in action_text:
            tags.append("missing_final_answer")
    elif status == "task limit reached":
        tags.append("task_limit")

    if is_mutation:
        tags.append("mutation_task")
        if "Action: Operation" not in action_text:
            tags.append("mutation_no_sql")
        else:
            has_select = any(
                "Action: Operation" in action and "SELECT " in action.upper()
                for action in agent_actions or []
            )
            if not has_select and "Action: Answer" in action_text:
                tags.append("mutation_unverified")
        if "Action: Answer" in action_text and final_answer not in ("", "[]"):
            tags.append("mutation_natural_language_answer")
    else:
        if query_type.startswith("aggregation-"):
            tags.append("aggregation_task")
        elif query_type in ("ranking", "comparison", "counting", "other"):
            tags.append(f"{query_type}_task")

        if any(phrase in final_lower for phrase in (
            "cannot answer", "can't answer", "not found", "not present",
            "not in the table", "need more information",
        )):
            tags.append("premature_not_found")

    return sorted(set(tags))


def _make_log_entry(sample: Dict, result: TaskClientOutput, is_correct: bool,
                    update_cycle: int, skill_snapshot: List[Dict]) -> Dict:
    history = []
    agent_actions = []
    if result.output and result.output.history:
        for msg in result.output.history:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            history.append({"role": role, "content": content})
            if role == "agent":
                agent_actions.append(content)

    failure_tags = [] if is_correct else _infer_failure_tags(sample, result, agent_actions)
    final_answer = _extract_final_answer_from_history(result.output.history) if result.output else ""
    task_result = dict(result.output.result) if result.output and result.output.result else None
    ground_truth = sample.get("answer")

    return {
        "sample_id": sample["id"],
        "instruction": sample.get("description", ""),
        "query_type": sample.get("type", "other"),
        "is_correct": is_correct,
        "update_cycle": update_cycle,
        "status": result.output.status if result.output else "error",
        "error": result.error,
        "failure_tags": failure_tags,
        "ground_truth": ground_truth,
        "task_result": task_result,
        "history_final_answer": final_answer,
        "agent_actions": agent_actions,
        "history": history,
        "skill_snapshot_before": skill_snapshot,
    }


class _TeeStream(io.TextIOBase):
    """Write to two streams simultaneously (e.g. stdout + log file)."""

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


class SkillCycleRunner:
    def __init__(self, config: Dict, run_dir: Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)

        cycle_cfg = config["cycle"]
        self.epochs: int = cycle_cfg["epochs"]
        self.update_every: int = cycle_cfg["update_every"]
        self.batch_concurrency: int = cycle_cfg.get("batch_concurrency", 5)
        self.max_proposals: int = cycle_cfg.get("max_proposals", 1)
        self.max_learned_skills: int = cycle_cfg.get("max_learned_skills", 10)
        self.grpo_k: int = cycle_cfg.get("grpo_k", 4)
        self.grpo_eval_n: int = cycle_cfg.get("grpo_eval_n", 20)
        self.run_baseline: bool = cycle_cfg.get("run_baseline", True)

        # Task-specific eval function loaded from config
        self._eval_fn = _load_eval_fn(config)

        task_cfg = config["task"]

        # Load splits — each sample has {"id": <task_index_str>, "description": "..."}
        self.dev_data = _load_required_json_list(Path(config["data"]["dev"]), "dev split")
        self.val_data = _load_required_json_list(Path(config["data"]["val"]), "val split")

        # For OS tasks, the task index IS the sample id (string key), so mapping is identity.
        # We keep _id_to_index for API consistency with the run logic.
        self._id_to_index: Dict[str, str] = {s["id"]: s["id"] for s in self.dev_data + self.val_data}

        # Task client (connects to already-running task worker)
        self.task_client = TaskClient(
            name=task_cfg["name"],
            controller_address=task_cfg["controller_address"],
        )

        # Skill repository (base read-only + per-run learned/)
        skills_cfg = config["skills"]
        self.skill_repo = SkillRepository(
            base_dir=Path(skills_cfg["base_dir"]),
            learned_dir=self.run_dir / "skills" / "learned",
        )

        # Build the base agent from config
        from src.typings.general import InstanceFactory
        base_agent = InstanceFactory(**config["agent"]).create()

        # Wrap base agent with skill injection
        self.skill_aware_agent = SkillAwareAgent(base_agent, self.skill_repo)

        # Updater uses the same base agent (not skill-aware — has its own prompt)
        self.updater = SkillUpdater(base_agent, max_proposals=self.max_proposals,
                                    max_learned_skills=self.max_learned_skills)

        self._val_scores_path = self.run_dir / "val_scores.json"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        log_path = self.run_dir / "run.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        original_stdout = sys.stdout
        sys.stdout = _TeeStream(original_stdout, log_file)
        try:
            self._run_inner()
        finally:
            sys.stdout = original_stdout
            log_file.close()

    def _run_inner(self) -> None:
        if self.run_baseline:
            print(f"\n{'='*60}")
            print(f"  BASELINE (before epoch 0)")
            print(f"{'='*60}")
            baseline_dir = self.run_dir / "baseline"
            baseline_dir.mkdir(exist_ok=True)
            baseline_score = self._evaluate_val(epoch="baseline", epoch_dir=baseline_dir)
            print(f"[Baseline] Val: {baseline_score:.1%}")

        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"  EPOCH {epoch}")
            print(f"{'='*60}")
            self._run_epoch(epoch)

        print("\n[SkillCycle] Training complete.")
        self._print_learning_curve()

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------

    def _load_prev_results(self, epoch: int) -> Optional[Dict[str, bool]]:
        if epoch == 0:
            return None
        prev_path = self.run_dir / f"epoch_{epoch - 1}" / "dev_runs.jsonl"
        if not prev_path.exists():
            return None
        results = {}
        with open(prev_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    results[entry["sample_id"]] = entry["is_correct"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return results

    def _run_epoch(self, epoch: int) -> None:
        epoch_dir = self.run_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        dev_runs_path = epoch_dir / "dev_runs.jsonl"
        updates_path = epoch_dir / "skill_updates.json"

        prev_results = self._load_prev_results(epoch)

        rng = random.Random(epoch)
        dev = self.dev_data[:]
        rng.shuffle(dev)

        all_entries: List[Dict] = []
        update_events: List[Dict] = []
        update_cycle = 0

        batches = [
            dev[i: i + self.update_every]
            for i in range(0, len(dev), self.update_every)
        ]
        print(f"[Epoch {epoch}] {len(dev)} dev samples — "
              f"{len(batches)} batches of ≤{self.update_every}")

        for batch_idx, batch in enumerate(batches):
            print(f"\n  Batch {batch_idx} / {len(batches) - 1} "
                  f"(update_cycle={update_cycle}, {len(batch)} samples)")

            skill_snapshot = self.skill_repo.snapshot()
            entries = self._run_batch(batch, update_cycle, skill_snapshot)

            for e in entries:
                e["skill_snapshot_before"] = skill_snapshot

            all_entries.extend(entries)

            with open(dev_runs_path, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(_serialisable(e), ensure_ascii=False) + "\n")

            n_correct = sum(e["is_correct"] for e in entries)
            print(f"  Batch score: {n_correct}/{len(entries)}")

            print(f"  Running grouped proposal ranking update (K={self.grpo_k})...")
            applied, grpo_log, raw_proposals = self._grpo_skill_update(
                current_entries=entries,
                all_entries=all_entries,
                prev_results=prev_results,
                epoch=epoch,
                update_cycle=update_cycle,
            )

            for e in entries:
                e["updates_applied_after"] = applied

            event = {
                "epoch": epoch,
                "update_cycle": update_cycle,
                "batch_size": len(batch),
                "batch_correct": n_correct,
                "applied": applied,
                "raw_proposals": raw_proposals,
                "grpo": grpo_log,
            }
            update_events.append(event)
            update_cycle += 1

        with open(updates_path, "w", encoding="utf-8") as f:
            json.dump(update_events, f, indent=2, ensure_ascii=False)

        epoch_correct = sum(e["is_correct"] for e in all_entries)
        epoch_total = len(all_entries)
        dev_score = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        val_score = self._evaluate_val(epoch, epoch_dir, dev_score=dev_score)
        print(f"\n[Epoch {epoch}] Dev: {epoch_correct}/{epoch_total} "
              f"({epoch_correct/epoch_total:.1%}) | "
              f"Val: {val_score:.1%}")

    # ------------------------------------------------------------------
    # Grouped proposal ranking / best-of-K skill update
    # ------------------------------------------------------------------

    def _grpo_skill_update(
        self,
        current_entries: List[Dict],
        all_entries: List[Dict],
        prev_results: Optional[Dict[str, bool]],
        epoch: int,
        update_cycle: int,
    ):
        rng = random.Random(epoch * 1000 + update_cycle)

        half = self.grpo_eval_n // 2
        failing_ids = {e["sample_id"] for e in all_entries if not e["is_correct"]}
        passing_ids = {e["sample_id"] for e in all_entries if e["is_correct"]}

        id_to_sample = {s["id"]: s for s in self.dev_data}

        probe_failing = rng.sample(
            [id_to_sample[sid] for sid in failing_ids if sid in id_to_sample],
            min(half, len(failing_ids)),
        )
        probe_passing = rng.sample(
            [id_to_sample[sid] for sid in passing_ids if sid in id_to_sample],
            min(half, len(passing_ids)),
        )
        probe_set = probe_failing + probe_passing

        if not probe_set:
            print("  [ProposalRanking] no probe samples available, skipping update")
            return [], [], []

        print(f"  [ProposalRanking] probe set: {len(probe_failing)} failing + "
              f"{len(probe_passing)} passing = {len(probe_set)} samples")

        candidates = []
        all_raw_proposals = []
        for k in range(self.grpo_k):
            proposals = self.updater.propose(
                current_entries, self.skill_repo, prev_results=prev_results
            )
            all_raw_proposals.extend(proposals)
            validated = self.updater.validate(proposals, self.skill_repo)
            if validated:
                candidates.extend(validated)

        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c["action"], c["name"], c.get("content", "")[:100])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        print(f"  [ProposalRanking] {len(candidates)} proposals sampled, "
              f"{len(unique_candidates)} unique")

        if not unique_candidates:
            print("  [ProposalRanking] no valid proposals, skipping update")
            return [], [], all_raw_proposals

        def eval_candidate(proposal: Dict):
            forked = self.skill_repo.fork()
            try:
                self.updater.apply([proposal], forked)
                forked_agent = SkillAwareAgent(
                    self.skill_aware_agent.agent,
                    forked,
                )
                fixes = 0
                regressions = 0
                invalid_action_regressions = 0

                def run_probe(sample):
                    task_index = self._id_to_index[sample["id"]]
                    result = self.task_client.run_sample(task_index, forked_agent)
                    from src.client.task import TaskError
                    if result.error == TaskError.NOT_AVAILABLE.value:
                        return None
                    status = result.output.status if result.output else "error"
                    is_correct = _score_result(sample, result, self._eval_fn)
                    return is_correct, status

                with ThreadPoolExecutor(max_workers=self.batch_concurrency) as pool:
                    futures = {pool.submit(run_probe, s): s for s in probe_set}
                    for future in as_completed(futures):
                        sample = futures[future]
                        probe_result = future.result()
                        if probe_result is None:
                            continue
                        is_correct, status = probe_result
                        was_failing = sample["id"] in failing_ids
                        if was_failing and is_correct:
                            fixes += 1
                        elif not was_failing and not is_correct:
                            regressions += 1
                            if status == _INVALID_ACTION_STATUS:
                                invalid_action_regressions += 1

                score = (
                    fixes
                    - regressions
                    - (_INVALID_ACTION_REGRESSION_PENALTY - 1) * invalid_action_regressions
                )
                return score, fixes, regressions, invalid_action_regressions
            finally:
                forked.cleanup()

        grpo_log = []
        best_score = 0
        best_candidate = None
        best_stats = (0, 0, 0)

        for proposal in unique_candidates:
            try:
                score, fixes, regressions, invalid_action_regressions = eval_candidate(proposal)
            except Exception as e:
                print(f"  [ProposalRanking] eval_candidate failed for "
                      f"{proposal.get('action')}::{proposal.get('name')}: {e}")
                continue
            action = proposal["action"]
            name = proposal["name"]
            print(
                f"  [ProposalRanking] {action}::{name} → score={score:+d} "
                f"(fixes={fixes}, regressions={regressions}, "
                f"invalid_action_regressions={invalid_action_regressions})"
            )
            grpo_log.append({
                "proposal": proposal,
                "net": score,
                "score": score,
                "fixes": fixes,
                "regressions": regressions,
                "invalid_action_regressions": invalid_action_regressions,
                "invalid_action_regression_penalty": _INVALID_ACTION_REGRESSION_PENALTY,
            })
            if score > best_score:
                best_score = score
                best_candidate = proposal
                best_stats = (fixes, regressions, invalid_action_regressions)

        if best_candidate is None:
            print("  [ProposalRanking] no proposal improved weighted score — applying nothing")
            return [], grpo_log, all_raw_proposals

        print(f"  [ProposalRanking] winner: {best_candidate['action']}::{best_candidate['name']} "
              f"score={best_score:+d} (fixes={best_stats[0]}, "
              f"regressions={best_stats[1]}, invalid_action_regressions={best_stats[2]})")
        applied = self.updater.apply([best_candidate], self.skill_repo)
        return applied, grpo_log, all_raw_proposals

    # ------------------------------------------------------------------
    # Batch execution (parallel)
    # ------------------------------------------------------------------

    def _run_batch(self, batch: List[Dict], update_cycle: int,
                   skill_snapshot: List[Dict]) -> List[Dict]:
        entries = [None] * len(batch)

        def run_one(idx: int, sample: Dict):
            return idx, self._run_single(sample)

        with ThreadPoolExecutor(max_workers=self.batch_concurrency) as pool:
            futures = {pool.submit(run_one, i, s): i for i, s in enumerate(batch)}
            for future in as_completed(futures):
                idx, (result, is_correct) = future.result()
                entries[idx] = _make_log_entry(
                    batch[idx], result, is_correct, update_cycle, skill_snapshot,
                )

        return entries

    def _run_single(self, sample: Dict, max_retries: int = 3):
        task_index = self._id_to_index[sample["id"]]
        for attempt in range(max_retries):
            result: TaskClientOutput = self.task_client.run_sample(
                task_index, self.skill_aware_agent
            )
            from src.client.task import TaskError
            if result.error == TaskError.NOT_AVAILABLE.value:
                wait = 5 * (attempt + 1)
                print(f"[SkillCycle] {sample['id']} not available, retry in {wait}s")
                time.sleep(wait)
                continue
            break
        is_correct = _score_result(sample, result, self._eval_fn)
        return result, is_correct

    # ------------------------------------------------------------------
    # Val evaluation
    # ------------------------------------------------------------------

    def _evaluate_val(self, epoch, epoch_dir: Path, dev_score: float = None) -> float:
        print(f"\n  [Val] evaluating {len(self.val_data)} samples...")
        correct = 0
        total = len(self.val_data)

        val_entries = [None] * total

        def run_one(idx: int, sample: Dict):
            task_index = self._id_to_index[sample["id"]]
            from src.client.task import TaskError
            for attempt in range(3):
                result = self.task_client.run_sample(task_index, self.skill_aware_agent)
                if result.error != TaskError.NOT_AVAILABLE.value:
                    break
                time.sleep(5 * (attempt + 1))
            is_correct = _score_result(sample, result, self._eval_fn)
            return idx, is_correct

        with ThreadPoolExecutor(max_workers=self.batch_concurrency) as pool:
            futures = {pool.submit(run_one, i, s): i
                       for i, s in enumerate(self.val_data)}
            for future in as_completed(futures):
                idx, is_correct = future.result()
                val_entries[idx] = {"sample_id": self.val_data[idx]["id"],
                                    "is_correct": is_correct}
                if is_correct:
                    correct += 1

        score = correct / total if total > 0 else 0.0

        val_score_record = {"epoch": epoch, "score": score,
                            "n_correct": correct, "n_total": total,
                            "dev_score": dev_score}
        with open(epoch_dir / "val_score.json", "w") as f:
            json.dump(val_score_record, f, indent=2)

        curve = _load_json_list_or_empty(self._val_scores_path, "val learning curve")
        curve.append(val_score_record)
        with open(self._val_scores_path, "w") as f:
            json.dump(curve, f, indent=2)

        return score

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_learning_curve(self) -> None:
        if not self._val_scores_path.exists():
            return
        curve = _load_json_list_or_empty(self._val_scores_path, "val learning curve")
        if not curve:
            return
        print("\nVal learning curve:")
        for entry in curve:
            bar = "█" * int(entry["score"] * 20)
            label = f"{entry['epoch']:>8}" if isinstance(entry["epoch"], int) else f"{'baseline':>8}"
            print(f"  {label}: {entry['score']:.1%}  {bar}")


# ------------------------------------------------------------------
# Serialisation helper
# ------------------------------------------------------------------

def _serialisable(obj: Any) -> Any:
    """Recursively convert Pydantic models / non-JSON-native types."""
    if hasattr(obj, "dict"):
        return _serialisable(obj.dict())
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    return obj
