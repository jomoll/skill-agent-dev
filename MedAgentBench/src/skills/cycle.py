"""
SkillCycleRunner — the main training loop for autonomous skill learning.

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

import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.client.agents.skill_aware_agent import SkillAwareAgent
from src.client.task import TaskClient
from src.skills.repository import SkillRepository
from src.skills.updater import SkillUpdater
from src.typings import TaskClientOutput

try:
    from src.server.tasks.medagentbench.eval import eval as _medagent_eval
    _EVAL_AVAILABLE = True
except Exception:
    _EVAL_AVAILABLE = False
    print("[SkillCycle] Warning: could not import eval — scores will be None")


def _score_result(sample: Dict, result: TaskClientOutput, fhir_api_base: str) -> bool:
    if not _EVAL_AVAILABLE:
        return False
    if result.error or result.output is None:
        return False
    if result.output.result is None:
        return False
    try:
        return _medagent_eval(sample, result.output, fhir_api_base) is True
    except Exception as e:
        print(f"[SkillCycle] eval error for {sample.get('id')}: {e}")
        return False


_VERIFIABLE_RESOURCES = {
    "Observation":       "patient",
    "MedicationRequest": "patient",
    "ServiceRequest":    "patient",
}
_POST_ACCEPTED_PREFIX = "POST request accepted and executed successfully"
_INVALID_ACTION_STATUS = "agent invalid action"
_INVALID_ACTION_REGRESSION_PENALTY = 2


def _compute_skill_effectiveness(
    all_entries: List[Dict],
    prev_results: Optional[Dict[str, bool]],
) -> Dict[str, Any]:
    """
    For each learned skill, count how many times it was present when a sample
    changed state relative to the previous epoch.
      fix:        was failing → now passing
      regression: was passing → now failing
    """
    stats: Dict[str, Any] = {}
    for entry in all_entries:
        sample_id = entry.get("sample_id")
        is_correct = entry.get("is_correct", False)
        snapshot = entry.get("skill_snapshot_before") or []
        skill_names = [s["name"] for s in snapshot if s["name"] != "skeleton"]
        prev = prev_results.get(sample_id) if prev_results and sample_id else None
        for skill_name in skill_names:
            if skill_name not in stats:
                stats[skill_name] = {"fixes": 0, "regressions": 0, "runs": 0}
            stats[skill_name]["runs"] += 1
            if prev is not None:
                if not prev and is_correct:
                    stats[skill_name]["fixes"] += 1
                elif prev and not is_correct:
                    stats[skill_name]["regressions"] += 1
    return stats


def _load_required_json_list(path: Path, label: str) -> List[Dict]:
    """
    Load a required JSON array artifact with a useful error message.

    Unlike append-only logs such as val_scores.json, dataset files are required
    inputs. If one is missing, empty, invalid, or not a JSON array, raise a
    clear ValueError instead of surfacing a low-level JSONDecodeError.
    """
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
    """
    Best-effort loader for append-only JSON list artifacts.

    If the file is missing, empty, or truncated/corrupt (for example after an
    interrupted run), return an empty list instead of crashing the whole cycle.
    """
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


def _collect_post_verifications(history: List[Dict], fhir_api_base: str) -> List[Dict]:
    """
    Called immediately after run_sample() returns, while the task's FHIR state
    is still live.  Scans the finished history for POST+accepted pairs, GETs
    each stored resource, and returns a list of verification dicts:
        {"after_turn": int, "note": str}
    where after_turn is the index of the POST-accepted user message.

    Results are stored alongside the log entry and injected into the trace
    shown to the skill updater — the live agent never sees them.
    """
    import re
    import requests

    verifications = []
    for i in range(len(history) - 1):
        msg = history[i]
        next_msg = history[i + 1]
        if not (msg.get("role") == "agent"
                and msg.get("content", "").strip().startswith("POST")
                and next_msg.get("role") == "user"
                and next_msg.get("content", "").startswith(_POST_ACCEPTED_PREFIX)):
            continue

        content = msg["content"].strip()
        lines = content.split("\n", 1)
        post_url = lines[0][4:].strip()  # strip "POST "

        m = re.search(r"/fhir/(\w+)", post_url)
        if not m or m.group(1) not in _VERIFIABLE_RESOURCES:
            continue
        resource_type = m.group(1)
        patient_param = _VERIFIABLE_RESOURCES[resource_type]

        # Use fhir_api_base from config (authoritative) rather than parsing
        # the URL from the POST line, which may differ in scheme/host
        fhir_base = fhir_api_base.rstrip("/") + "/"

        mrn = None
        if len(lines) > 1:
            try:
                body = json.loads(lines[1])
                ref = (body.get("subject", {}).get("reference", "")
                       or body.get("patient", {}).get("reference", ""))
                mrn = ref.split("/")[-1] if ref else None
            except (json.JSONDecodeError, AttributeError):
                pass

        if not mrn:
            continue

        try:
            params = {
                patient_param: mrn,
                "_sort": "-_lastUpdated",
                "_count": "1",
                "_format": "json",
            }
            resp = requests.get(
                f"{fhir_base}{resource_type}", params=params, timeout=30
            )
            if resp.status_code == 200:
                entries = resp.json().get("entry", [])
                if entries:
                    resource = entries[0].get("resource", {})
                    note = (
                        f"[POST verification — log only, not seen by agent] "
                        f"The {resource_type} was stored as:\n"
                        f"{json.dumps(resource, indent=2)}"
                    )
                else:
                    note = (
                        f"[POST verification — log only] Warning: "
                        f"{resource_type} POST accepted but resource not found "
                        f"on retrieval — may not have been stored correctly."
                    )
            else:
                note = (
                    f"[POST verification — log only] GET returned "
                    f"HTTP {resp.status_code}."
                )
        except Exception as e:
            note = f"[POST verification — log only] GET failed: {e}"

        # i+1 is the POST-accepted message; insert note after it
        verifications.append({"after_turn": i + 1, "note": note})

    return verifications


def _apply_verifications(history: List[Dict],
                         verifications: List[Dict]) -> List[Dict]:
    """
    Merge pre-fetched verification notes back into a history copy as
    system_note entries, inserting them right after the POST-accepted messages.
    Offsets are adjusted as notes are inserted.
    """
    enriched = list(history)
    offset = 0
    for v in sorted(verifications, key=lambda x: x["after_turn"]):
        insert_at = v["after_turn"] + 1 + offset
        enriched.insert(insert_at, {"role": "system_note", "content": v["note"]})
        offset += 1
    return enriched


def _make_log_entry(sample: Dict, result: TaskClientOutput, is_correct: bool,
                    update_cycle: int, skill_snapshot: List[Dict],
                    post_verifications: Optional[List[Dict]] = None) -> Dict:
    history = []
    agent_actions = []
    if result.output and result.output.history:
        for msg in result.output.history:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            history.append({"role": role, "content": content})
            if role == "agent":
                agent_actions.append(content)

    # Inject pre-fetched GET-after-POST notes (log only, never sent to agent)
    if post_verifications:
        history = _apply_verifications(history, post_verifications)

    return {
        "sample_id": sample["id"],
        "instruction": sample["instruction"],
        "is_correct": is_correct,
        "update_cycle": update_cycle,
        "status": result.output.status if result.output else "error",
        "error": result.error,
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
        self.grpo_eval_n: int = cycle_cfg.get("grpo_eval_n", 20)  # 10 pass + 10 fail
        self.run_baseline: bool = cycle_cfg.get("run_baseline", True)

        task_cfg = config["task"]
        self.fhir_api_base: str = task_cfg["fhir_api_base"]

        # Build id → original dataset index mapping
        full_data_path = Path(config["data"]["full"])
        full_data: List[Dict] = _load_required_json_list(full_data_path, "full dataset")
        self._id_to_index: Dict[str, int] = {s["id"]: i for i, s in enumerate(full_data)}

        # Load splits
        self.dev_data = _load_required_json_list(Path(config["data"]["dev"]), "dev split")
        self.val_data = _load_required_json_list(Path(config["data"]["val"]), "val split")

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

        # Wrap base agent with skill injection only — execution chain is clean
        self.skill_aware_agent = SkillAwareAgent(base_agent, self.skill_repo)

        # Updater uses the same base agent (not skill-aware — updater has its own prompt)
        self.updater = SkillUpdater(base_agent, max_proposals=self.max_proposals,
                                    max_learned_skills=self.max_learned_skills)

        # Val learning-curve log
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
        """Load sample_id → is_correct from the previous epoch's dev_runs.jsonl."""
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

        # Shuffle dev set with fixed seed per epoch
        rng = random.Random(epoch)
        dev = self.dev_data[:]
        rng.shuffle(dev)

        all_entries: List[Dict] = []   # accumulated across entire epoch
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

            # Snapshot of learned skills before this batch
            skill_snapshot = self.skill_repo.snapshot()

            # Run batch in parallel
            entries = self._run_batch(batch, update_cycle, skill_snapshot)

            # Append update_cycle and snapshot to each entry (for updater)
            for e in entries:
                e["skill_snapshot_before"] = skill_snapshot

            all_entries.extend(entries)

            # Write to dev_runs.jsonl incrementally
            with open(dev_runs_path, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(_serialisable(e), ensure_ascii=False) + "\n")

            n_correct = sum(e["is_correct"] for e in entries)
            print(f"  Batch score: {n_correct}/{len(entries)}")

            # Grouped proposal ranking / best-of-K skill update
            print(f"  Running grouped proposal ranking update (K={self.grpo_k})...")
            applied, grpo_log, raw_proposals = self._grpo_skill_update(
                current_entries=entries,
                all_entries=all_entries,
                prev_results=prev_results,
                epoch=epoch,
                update_cycle=update_cycle,
            )

            # Annotate the last batch's entries with what was applied
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

        # Write skill update log for this epoch
        with open(updates_path, "w", encoding="utf-8") as f:
            json.dump(update_events, f, indent=2, ensure_ascii=False)

        # Silent val evaluation
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
        """
        Sample K single-change proposals, evaluate each on a balanced probe set,
        apply the one with the best weighted score. Regressions caused by
        `agent invalid action` are penalized more strongly than ordinary
        regressions. If no proposal improves on the baseline (score ≤ 0),
        nothing is applied.

        Returns (applied: List[Dict], grpo_log: List[Dict]).
        """
        rng = random.Random(epoch * 1000 + update_cycle)
        id_to_sample = {s["id"]: s for s in self.dev_data}

        probe_set, probe_failing_ids = self._build_probe_set(
            all_entries, prev_results, epoch, update_cycle, id_to_sample, rng
        )
        if probe_set is None:
            print("  [ProposalRanking] skipping update: no out-of-sample probe data "
                  "(epoch 0, batch 0)")
            return [], [], []

        n_failing = sum(1 for s in probe_set if s["id"] in probe_failing_ids)
        n_passing = len(probe_set) - n_failing
        print(f"  [ProposalRanking] probe set: {n_failing} failing + "
              f"{n_passing} passing = {len(probe_set)} samples "
              f"(out-of-sample from prior batches)")

        # Sample candidate edits from the current batch only. The updater may
        # return multiple possible single-edit proposals per call; each
        # validated edit becomes its own candidate and is ranked independently.
        # Candidate evaluation still uses the probe set built below.
        skill_effectiveness = _compute_skill_effectiveness(all_entries, prev_results)

        candidates = []
        all_raw_proposals = []  # collect every proposal before validation
        for k in range(self.grpo_k):
            proposals = self.updater.propose(
                current_entries, self.skill_repo, prev_results=prev_results,
                skill_effectiveness=skill_effectiveness,
            )
            all_raw_proposals.extend(proposals)
            validated = self.updater.validate(proposals, self.skill_repo)
            if validated:
                candidates.extend(validated)

        # Deduplicate identical proposals
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

        # Evaluate each candidate on the probe set
        def eval_candidate(proposal: Dict):
            forked = self.skill_repo.fork()
            try:
                self.updater.apply([proposal], forked)
                forked_agent = SkillAwareAgent(
                    # reuse the same base agent — only the repo differs
                    self.skill_aware_agent.agent,
                    forked,
                )
                fixes = 0
                regressions = 0
                invalid_action_regressions = 0

                def run_probe(sample):
                    original_index = self._id_to_index[sample["id"]]
                    result = self.task_client.run_sample(original_index, forked_agent)
                    from src.client.task import TaskError
                    if result.error == TaskError.NOT_AVAILABLE.value:
                        return None
                    status = result.output.status if result.output else "error"
                    is_correct = _score_result(sample, result, self.fhir_api_base)
                    return is_correct, status

                with ThreadPoolExecutor(max_workers=self.batch_concurrency) as pool:
                    futures = {pool.submit(run_probe, s): s for s in probe_set}
                    for future in as_completed(futures):
                        sample = futures[future]
                        probe_result = future.result()
                        if probe_result is None:
                            continue
                        is_correct, status = probe_result
                        was_failing = sample["id"] in probe_failing_ids
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
        winner = dict(best_candidate)
        winner["_provenance"] = {
            "epoch": epoch,
            "update_cycle": update_cycle,
            "action": winner["action"],
            "probe_score": best_score,
            "fixes": best_stats[0],
            "regressions": best_stats[1],
            "triggering_sample_ids": [
                e["sample_id"] for e in current_entries if not e["is_correct"]
            ][:10],
        }
        applied = self.updater.apply([winner], self.skill_repo)
        return applied, grpo_log, all_raw_proposals

    # ------------------------------------------------------------------
    # Probe set construction
    # ------------------------------------------------------------------

    @staticmethod
    def _stratified_sample(
        samples: List[Dict],
        key_fn,
        n: int,
        rng: random.Random,
    ) -> List[Dict]:
        """Sample up to n items stratified by key_fn, with at least 1 per type."""
        if not samples or n <= 0:
            return []
        groups: Dict[str, List] = {}
        for s in samples:
            groups.setdefault(key_fn(s), []).append(s)
        per_type = max(1, n // len(groups))
        selected = []
        for group_items in groups.values():
            selected.extend(rng.sample(group_items, min(per_type, len(group_items))))
        remaining = n - len(selected)
        if remaining > 0:
            selected_ids = {id(s) for s in selected}
            pool = [s for s in samples if id(s) not in selected_ids]
            if pool:
                selected.extend(rng.sample(pool, min(remaining, len(pool))))
        return selected

    def _build_probe_set(
        self,
        all_entries: List[Dict],
        prev_results: Optional[Dict[str, bool]],
        epoch: int,
        update_cycle: int,
        id_to_sample: Dict[str, Dict],
        rng: random.Random,
    ):
        """
        Build a probe set that is out-of-sample relative to the generating batch.

        Priority:
          1. Entries from earlier batches of the current epoch
             (update_cycle < current update_cycle).
          2. For batch 0 of epoch > 0: use prev epoch results as the baseline.
          3. Epoch 0, batch 0 — no out-of-sample data exists → return (None, None)
             and the caller skips the update entirely.

        Returns (probe_samples, probe_failing_ids) or (None, None).
        """
        half = self.grpo_eval_n // 2
        type_key = lambda s: s.get("type", "other")

        prior_entries = [
            e for e in all_entries
            if e.get("update_cycle", update_cycle) < update_cycle
        ]
        if prior_entries:
            failing = [e for e in prior_entries if not e["is_correct"]]
            passing = [e for e in prior_entries if e["is_correct"]]
            probe_failing_ids = {e["sample_id"] for e in failing}
            probe = (
                self._stratified_sample(
                    [id_to_sample[e["sample_id"]] for e in failing if e["sample_id"] in id_to_sample],
                    type_key, half, rng,
                )
                + self._stratified_sample(
                    [id_to_sample[e["sample_id"]] for e in passing if e["sample_id"] in id_to_sample],
                    type_key, half, rng,
                )
            )
            return (probe, probe_failing_ids) if probe else (None, None)

        if epoch > 0 and prev_results:
            # Batch 0 of a later epoch: no current-epoch prior batches yet,
            # fall back to the previous epoch's results as the probe baseline.
            failing_ids = {sid for sid, ok in prev_results.items() if not ok}
            passing_ids = {sid for sid, ok in prev_results.items() if ok}
            probe = (
                self._stratified_sample(
                    [id_to_sample[sid] for sid in failing_ids if sid in id_to_sample],
                    type_key, half, rng,
                )
                + self._stratified_sample(
                    [id_to_sample[sid] for sid in passing_ids if sid in id_to_sample],
                    type_key, half, rng,
                )
            )
            return (probe, failing_ids) if probe else (None, None)

        return None, None  # epoch 0, batch 0 — skip update

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
                idx, (result, is_correct, post_verifications) = future.result()
                entries[idx] = _make_log_entry(
                    batch[idx], result, is_correct, update_cycle, skill_snapshot,
                    post_verifications=post_verifications,
                )

        return entries

    def _run_single(self, sample: Dict, max_retries: int = 3):
        original_index = self._id_to_index[sample["id"]]
        for attempt in range(max_retries):
            result: TaskClientOutput = self.task_client.run_sample(
                original_index, self.skill_aware_agent
            )
            from src.client.task import TaskError
            if result.error == TaskError.NOT_AVAILABLE.value:
                wait = 5 * (attempt + 1)
                print(f"[SkillCycle] {sample['id']} not available, retry in {wait}s")
                time.sleep(wait)
                continue
            break
        # Collect POST verifications immediately while the task's FHIR state is live
        raw_history = []
        if result.output and result.output.history:
            for msg in result.output.history:
                role = msg.role if hasattr(msg, "role") else msg["role"]
                content = msg.content if hasattr(msg, "content") else msg["content"]
                raw_history.append({"role": role, "content": content})
        post_verifications = _collect_post_verifications(raw_history, self.fhir_api_base)
        is_correct = _score_result(sample, result, self.fhir_api_base)
        return result, is_correct, post_verifications

    # ------------------------------------------------------------------
    # Val evaluation
    # ------------------------------------------------------------------

    def _evaluate_val(self, epoch, epoch_dir: Path, dev_score: float = None) -> float:
        print(f"\n  [Val] evaluating {len(self.val_data)} samples...")
        correct = 0
        total = len(self.val_data)

        val_entries = [None] * total

        def run_one(idx: int, sample: Dict):
            original_index = self._id_to_index[sample["id"]]
            from src.client.task import TaskError
            for attempt in range(3):
                result = self.task_client.run_sample(original_index, self.skill_aware_agent)
                if result.error != TaskError.NOT_AVAILABLE.value:
                    break
                time.sleep(5 * (attempt + 1))
            is_correct = _score_result(sample, result, self.fhir_api_base)
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

        # Write val score
        val_score_record = {"epoch": epoch, "score": score,
                            "n_correct": correct, "n_total": total,
                            "dev_score": dev_score}
        with open(epoch_dir / "val_score.json", "w") as f:
            json.dump(val_score_record, f, indent=2)

        # Append to learning curve
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
