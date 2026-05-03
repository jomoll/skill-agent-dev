"""
SkillAwareAgent — wraps any AgentClient and injects the current skill library
into the conversation history before each inference call.

Injection strategy depends on whether this is the first agent decision or a
continuation turn:

  First decision (no prior assistant/agent turn in history):
      Skills are PREPENDED to the last user message so the model reads them
      before the task instruction.  This interrupts reflexive first-action
      behaviour that would otherwise fire before any skill context is processed.

  Continuation turn (prior assistant/agent turn exists):
      Skills are APPENDED to the last user message (after the latest environment
      observation) so they remain at the recency-favoured end of the context,
      close to the generation point.

Each skill is introduced by its description so the model can judge applicability
at a glance; the full content follows for when the skill is relevant.

The model is told once, at the top of the block, to apply skills that match and
ignore the rest — no per-skill checklists.

Skills named "skeleton" (the read-only base template) are never injected.
If no skills exist and the task is not DBBench, history is passed through unchanged.
"""

import logging
import re
from typing import Any, Dict, FrozenSet, List

from ..agent import AgentClient
from src.skills.repository import SkillRepository

logger = logging.getLogger(__name__)

# Maximum skills injected per turn across all benchmarks.
_MAX_SKILLS = 3

_INSERT_RE = re.compile(
    r"\b(insert|was inserted|has been (added|inserted)"
    r"|add(?:ing)? a (new )?(row|record|entry)"
    r"|a new \w+(?: \w+)? (joined|hired|registered|added))\b",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"\b(update|was updated|has been (changed|updated|modified)"
    r"|modif(?:y|ied|ying)|change the (value|record|entry|salary|name|status))\b",
    re.IGNORECASE,
)


class SkillAwareAgent(AgentClient):
    def __init__(self, agent: AgentClient, skill_repo: SkillRepository) -> None:
        super().__init__()
        self.agent = agent
        self.skill_repo = skill_repo

    def inference(self, history: List[dict], tools=None):
        skills = [s for s in self.skill_repo.load_all() if s["name"] != "skeleton"]
        first_content = self._message_content(history[0]) if history else ""
        is_dbbench = self._is_dbbench_prompt(first_content)

        skills = self._select_skills(skills, first_content, is_dbbench)

        if not skills and not is_dbbench:
            return self._delegate(history, tools=tools)

        suffix_parts = []
        if skills:
            suffix_parts.append(self._render_skills(skills))
        if is_dbbench:
            suffix_parts.append(self._dbbench_protocol())

        skill_block = "\n\n" + "\n\n".join(suffix_parts)

        modified = [self._message_to_dict(message) for message in history]
        last_user_idx = max(
            (
                i for i, m in enumerate(modified)
                if m.get("role") in ("user", "system")
            ),
            default=0,
        )

        # Determine injection position based on whether the agent has already
        # taken at least one turn.  "agent" covers DBBench's acknowledgement
        # message; "assistant" covers standard chat-format tasks.
        is_first_decision = not any(
            m.get("role") in ("assistant", "agent")
            for m in modified[:last_user_idx + 1]
        )

        if is_first_decision:
            # Prepend: skills appear before the task instruction so the model
            # processes behavioural rules before reading the task and acting.
            new_content = (
                skill_block.lstrip("\n")
                + "\n\n"
                + (modified[last_user_idx].get("content") or "")
            )
        else:
            # Append: skills stay at the recency-favoured end of the context,
            # immediately before generation on continuation turns.
            new_content = (modified[last_user_idx].get("content") or "") + skill_block

        modified[last_user_idx] = dict(modified[last_user_idx], content=new_content)
        return self._delegate(modified, tools=tools)

    def _delegate(self, history: List[dict], tools=None):
        if tools is not None:
            try:
                return self.agent.inference(history, tools=tools)
            except TypeError:
                pass
        return self.agent.inference(history)

    @staticmethod
    def _message_to_dict(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if hasattr(message, "dict"):
            return message.dict(exclude_none=True)
        return {
            "role": getattr(message, "role", "user"),
            "content": getattr(message, "content", ""),
        }

    @classmethod
    def _message_content(cls, message: Any) -> str:
        item = cls._message_to_dict(message)
        content = item.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _render_skills(skills: list) -> str:
        header = (
            "---\n"
            "**Behavioral skills:** before each action, scan the skill descriptions "
            "below. If a skill's 'When to use' matches your current task or the action "
            "you are about to take, follow its guidance. Skip skills that do not match.\n"
        )
        blocks = []
        for s in skills:
            name = s["name"]
            desc = s.get("description", "")
            content = s.get("content", "")
            desc_line = f"*When to use: {desc}*\n" if desc else ""
            blocks.append(f"### {name}\n{desc_line}\n{content}")
        return header + "\n\n".join(blocks)

    @staticmethod
    def _skill_scope(skill: Dict) -> FrozenSet[str]:
        """Map a skill's tags to the query types it applies to.

        A skill with no query-type tags is treated as general and matches all
        types — it competes for a slot under the cap rather than being excluded.
        """
        tags = {t.lower() for t in (skill.get("tags") or [])}
        scope: set = set()
        if tags & {"insert"}:
            scope.add("INSERT")
        if tags & {"update", "delete"}:
            scope.add("UPDATE")
        if tags & {"mutation"}:
            scope |= {"INSERT", "UPDATE"}
        if tags & {"select", "retrieval", "read"}:
            scope.add("READ")
        return frozenset(scope) if scope else frozenset({"INSERT", "UPDATE", "READ"})

    @classmethod
    def _select_skills(
        cls, skills: List[Dict], task_text: str, is_dbbench: bool
    ) -> List[Dict]:
        """Return at most _MAX_SKILLS skills ranked by relevance to the current task.

        For DBBench, query type is inferred from the task instruction and used to
        exclude skills whose tags declare them for a different query type (e.g. an
        INSERT skill is dropped on a read task).  For other benchmarks no query-type
        inference is attempted, so only the cap and scope-specificity ranking apply.

        Skills with narrower scope (fewer covered query types) rank higher, so a
        targeted INSERT-only skill beats a general mutation skill beats an untagged
        skill when all three are eligible.
        """
        if is_dbbench:
            if _INSERT_RE.search(task_text):
                query_type = "INSERT"
            elif _UPDATE_RE.search(task_text):
                query_type = "UPDATE"
            else:
                query_type = "READ"
        else:
            query_type = None

        ranked: List[tuple] = []
        excluded: List[str] = []
        for skill in skills:
            scope = cls._skill_scope(skill)
            if query_type is None or query_type in scope:
                ranked.append((len(scope), skill))  # smaller scope → higher priority
            else:
                excluded.append(skill["name"])

        ranked.sort(key=lambda x: x[0])
        selected = [s for _, s in ranked[:_MAX_SKILLS]]

        logger.info(
            "skill_selection query_type=%s selected=%s excluded=%s",
            query_type or "none",
            [s["name"] for s in selected],
            excluded,
        )
        return selected

    @classmethod
    def _is_dbbench_prompt(cls, content: str) -> bool:
        if not isinstance(content, str):
            return False
        text = content.lower()
        return (
            "help me operate a mysql database with sql" in text
            or "execute_sql" in text
            or "commit_final_answer" in text
        )

    @staticmethod
    def _dbbench_protocol() -> str:
        return (
            "---\n"
            "**SQL tool reminder:** use the provided DBBench tools; do not write "
            "tool invocations as plain text.\n"
            "- Query or mutate the database with `execute_sql`.\n"
            "- Submit the answer with `commit_final_answer` only when done.\n"
            "- For INSERT/UPDATE/DELETE tasks, verify the changed rows with a "
            "targeted SELECT before submitting."
        )
