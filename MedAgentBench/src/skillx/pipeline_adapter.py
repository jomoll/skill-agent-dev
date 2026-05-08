from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "successful_trajectory": normalised,
        "plan": f"# api step 1: {instruction}",
        "exp_metadata": {},
        "reward": 1.0,
    }


def _collect_skill_dicts(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten plan_step_metadata into a list of raw skill dicts."""
    skills: List[Dict[str, Any]] = []
    for step_skills in (item.get("plan_step_metadata") or {}).values():
        for entry in step_skills:
            if not isinstance(entry, dict):
                continue
            option = entry.get("option", "add")
            if option not in ("add", "modify"):
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


class SkillXPipelineAdapter:
    """
    Orchestrates SkillX extraction → filter → library merge for one epoch.

    Imports SkillX submodules lazily so that the langchain-dependent pipeline.py
    is never loaded.
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

        # skillx_dir should be the SkillX repo root (e.g. /path/to/SkillX).
        # We add its parent to sys.path so that `import SkillX.extraction` works
        # with relative imports intact.
        skillx_dir = config.get("skillx_dir", "")
        if skillx_dir:
            parent = str(Path(skillx_dir).parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

        self._import_skillx()

        self.library = self._load_library()
        self._epoch = 0

    def _import_skillx(self) -> None:
        from SkillX.extraction.skill_extractor import FunctionalSkillExtractor
        from SkillX.filtering.pipeline import TwoStageFilterPipeline
        from SkillX.core.skill import Skill, SkillLibrary

        self._FunctionalSkillExtractor = FunctionalSkillExtractor
        self._TwoStageFilterPipeline = TwoStageFilterPipeline
        self._Skill = Skill
        self._SkillLibrary = SkillLibrary

        self.extractor = FunctionalSkillExtractor(
            llm=self.lm_adapter,
            benchmark="medagentbench",
            verbose=False,
        )
        self.filter_pipeline = TwoStageFilterPipeline(
            llm=self.lm_adapter,
            benchmark="medagentbench",
            skip_stage1=not self.config.get("filter_stage1", True),
            skip_stage2=True,
            verbose=False,
        )

    def _load_library(self) -> Any:
        if self.library_path.exists():
            try:
                return self._SkillLibrary.load(str(self.library_path))
            except Exception as exc:
                logger.warning("Failed to load skill library, starting fresh: %s", exc)
        return self._SkillLibrary(benchmark="medagentbench")

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

        # Extract skills from each trajectory
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
                "n_after_merge": len(self.library.functional),
            }

        # Filter
        filtered_skills = raw_skills
        if self.config.get("filter_stage1", True):
            try:
                filtered_skills = await self.filter_pipeline.filter(
                    raw_skills, batch_size=10, max_concurrent=3, show_progress=False
                )
            except Exception as exc:
                logger.warning("Filtering failed, using unfiltered skills: %s", exc)

        n_filtered = len(filtered_skills)

        # Convert to Skill objects and update library
        skill_objects = []
        for skill_data in filtered_skills:
            try:
                skill_objects.append(self._Skill.from_dict(skill_data))
            except Exception as exc:
                logger.debug("Skipping invalid skill dict: %s", exc)

        if skill_objects:
            self.library.merge(skill_objects, epoch=self._epoch)

        return {
            "n_successful": len(trajectories),
            "n_extracted": n_extracted,
            "n_filtered": n_filtered,
            "n_after_merge": len(self.library.functional),
        }

    def get_skills(self) -> List[Any]:
        """Return all functional skills in the library."""
        return list(self.library.functional)
