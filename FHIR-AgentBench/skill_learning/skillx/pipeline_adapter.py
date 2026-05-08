from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _entry_to_trajectory(entry: Dict[str, Any]) -> Dict[str, Any]:
    instruction = entry.get("instruction") or entry.get("sample_id") or ""
    history = entry.get("history") or []
    normalised = []
    for m in history:
        role = m.get("role", "user")
        if role == "agent":
            role = "assistant"
        normalised.append({"role": role, "content": str(m.get("content", ""))})
    return {
        "trajectory_id": entry.get("sample_id", ""),
        "task_id": entry.get("sample_id", ""),
        "user_task": instruction,
        "task_history": normalised,          # PlanExtractor reads task_history
        "successful_trajectory": normalised,  # FunctionalSkillExtractor reads this
        "plan": f"# api step 1: {instruction}",  # fallback stub
        "exp_metadata": {},
        "reward": 1.0,
    }


def _collect_skill_dicts(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    skills: List[Dict[str, Any]] = []
    for step_skills in (item.get("plan_step_metadata") or {}).values():
        for entry in step_skills:
            if not isinstance(entry, dict):
                continue
            if entry.get("option", "add") not in ("add", "modify"):
                continue
            skill_data = entry.get("skill", entry)
            if not isinstance(skill_data, dict) or "name" not in skill_data:
                continue
            skill_data.setdefault("document", "")
            skill_data.setdefault("content", "")
            skill_data.setdefault("tools", [])
            skill_data.setdefault("metadata", {})
            skill_data["metadata"].setdefault("skill_type", "functional")
            skills.append(skill_data)
    return skills


def _group_by_name(skills: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in skills:
        key = (s.get("name") or "").strip().lower()
        groups[key].append(s)
    return list(groups.values())


class SkillXPipelineAdapter:
    """
    Orchestrates SkillX extraction → filter → LLM-merge → library merge for one epoch.

    Uses upstream FunctionalSkillExtractor, PlanExtractor (optional),
    TwoStageFilterPipeline (stage 1 only), and SkillMerger (name-based LLM dedup).
    Imports SkillX submodules directly to avoid the langchain-dependent pipeline.py.
    """

    def __init__(
        self,
        lm_adapter: Any,
        library_path: Path,
        config: Dict[str, Any],
    ) -> None:
        self.lm_adapter = lm_adapter
        self.library_path = Path(library_path)
        self.config = config

        self._import_skillx()
        self.library = self._load_library()
        self._epoch = 0

    def _import_skillx(self) -> None:
        from .vendor.extractor import PlanExtractor, FunctionalSkillExtractor
        from .vendor.filter import TwoStageFilterPipeline
        from .vendor.merger import SkillMerger
        from .vendor.skill import Skill, SkillLibrary

        self._Skill = Skill
        self._SkillLibrary = SkillLibrary

        self.plan_extractor = PlanExtractor(llm=self.lm_adapter)
        self.extractor = FunctionalSkillExtractor(llm=self.lm_adapter)
        self.filter_pipeline = TwoStageFilterPipeline(
            llm=self.lm_adapter,
            skip_stage1=not self.config.get("filter_stage1", True),
        )
        self.skill_merger = SkillMerger(llm=self.lm_adapter)

    def _load_library(self) -> Any:
        if self.library_path.exists():
            try:
                return self._SkillLibrary.load(str(self.library_path))
            except Exception as exc:
                logger.warning("Failed to load skill library, starting fresh: %s", exc)
        return self._SkillLibrary(benchmark="appworld")

    def _save_library(self) -> None:
        self.library.save(str(self.library_path))

    def run_epoch(self, dev_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [
            e for e in dev_entries
            if e.get("is_correct") and not e.get("error")
        ]
        if not successful:
            return {"n_successful": 0, "n_extracted": 0, "n_after_merge": len(self.library.functional)}

        trajectories = [_entry_to_trajectory(e) for e in successful]
        stats = asyncio.run(self._extract_and_update(trajectories))
        self._save_library()
        return stats

    async def _extract_and_update(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._epoch += 1

        # Step 1: extract real plans (PlanExtractor), fall back to stub plan
        if self.config.get("use_plan_extraction", True):
            planned = []
            for traj in trajectories:
                try:
                    result = await self.plan_extractor.extract(traj)
                    planned.append(result if result and result.get("plan") else traj)
                except Exception as exc:
                    logger.debug("Plan extraction failed for %s: %s", traj.get("task_id"), exc)
                    planned.append(traj)
            trajectories = planned

        # Step 2: extract functional skills from each trajectory
        raw_skills: List[Dict[str, Any]] = []
        for traj in trajectories:
            try:
                result = await self.extractor.extract(traj)
                if result:
                    raw_skills.extend(_collect_skill_dicts(result))
            except Exception as exc:
                logger.warning("Skill extraction failed for %s: %s", traj.get("task_id"), exc)

        n_extracted = len(raw_skills)
        if not raw_skills:
            return {
                "n_successful": len(trajectories),
                "n_extracted": 0,
                "n_filtered": 0,
                "n_merged": 0,
                "n_after_merge": len(self.library.functional),
            }

        # Step 3: filter (stage 1 quality filter; stage 2 skipped — no tool schemas)
        filtered_skills = raw_skills
        if self.config.get("filter_stage1", True):
            try:
                filtered_skills = await self.filter_pipeline.filter(
                    raw_skills, batch_size=10, max_concurrent=3, show_progress=False
                )
            except Exception as exc:
                logger.warning("Filtering failed, using unfiltered skills: %s", exc)

        n_filtered = len(filtered_skills)

        # Step 4: LLM-merge skills with the same name (upstream SkillMerger, no embeddings)
        merged_skills = filtered_skills
        if self.config.get("enable_llm_merge", True) and filtered_skills:
            clusters = _group_by_name(filtered_skills)
            multi = [c for c in clusters if len(c) > 1]
            single = [c[0] for c in clusters if len(c) == 1]
            if multi:
                try:
                    merge_results = await self.skill_merger.merge_clusters(multi)
                    merged_skills = single + [r for r in merge_results if r]
                except Exception as exc:
                    logger.warning("LLM merge failed, using unmerged skills: %s", exc)
            else:
                merged_skills = single

        n_merged = len(merged_skills)

        # Step 5: convert to Skill objects and update library
        skill_objects = []
        for skill_data in merged_skills:
            inner = skill_data.get("skill", skill_data)
            if not isinstance(inner, dict):
                continue
            inner.setdefault("name", skill_data.get("name", ""))
            inner.setdefault("document", "")
            inner.setdefault("content", "")
            inner.setdefault("tools", [])
            inner.setdefault("metadata", {})
            inner["metadata"].setdefault("skill_type", "functional")
            try:
                skill_objects.append(self._Skill.from_dict(inner))
            except Exception as exc:
                logger.debug("Skipping invalid skill dict: %s", exc)

        if skill_objects:
            self.library.merge(skill_objects, epoch=self._epoch)

        return {
            "n_successful": len(trajectories),
            "n_extracted": n_extracted,
            "n_filtered": n_filtered,
            "n_merged": n_merged,
            "n_after_merge": len(self.library.functional),
        }

    def get_skills(self) -> List[Any]:
        return list(self.library.functional)

    def build_skill_block(self, query: str, top_k: int = 5) -> str:
        """Build skill injection block for a given query."""
        if not self.library_path.exists():
            return ""
        try:
            with self.library_path.open(encoding="utf-8") as f:
                data = json.load(f)
            skills = data.get("skills", {}).get("functional", [])
        except Exception:
            return ""
        if not skills:
            return ""

        def _tokenize(text: str) -> List[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        def _overlap(q: str, d: str) -> float:
            qs = set(_tokenize(q))
            ds = set(_tokenize(d))
            if not qs or not ds:
                return 0.0
            return len(qs & ds) / len(qs | ds)

        scored = [
            (_overlap(query, f"{s.get('name', '')} {s.get('document', '')}"), s)
            for s in skills
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for sc, s in scored[:top_k] if scored[0][0] > 0]
        if not top:
            return ""

        parts = ["<skillx_memory>\nBehavioral skills extracted from past experience:\n"]
        for s in top:
            parts.append(f"### {s['name']}\n{s['document']}\n\n```\n{s['content']}\n```\n")
        parts.append("</skillx_memory>")
        return "\n".join(parts)
