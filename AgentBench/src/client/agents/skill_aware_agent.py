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

from typing import List

from ..agent import AgentClient
from src.skills.repository import SkillRepository


class SkillAwareAgent(AgentClient):
    def __init__(self, agent: AgentClient, skill_repo: SkillRepository) -> None:
        super().__init__()
        self.agent = agent
        self.skill_repo = skill_repo

    def inference(self, history: List[dict]) -> str:
        skills = [s for s in self.skill_repo.load_all() if s["name"] != "skeleton"]
        first_content = history[0]["content"]
        is_dbbench = self._is_dbbench_prompt(first_content)

        if not skills and not is_dbbench:
            return self.agent.inference(history)

        suffix_parts = []
        if skills:
            suffix_parts.append(self._render_skills(skills))
        if is_dbbench:
            suffix_parts.append(self._dbbench_protocol())

        skill_block = "\n\n" + "\n\n".join(suffix_parts)

        modified = list(history)
        last_user_idx = max(
            (i for i, m in enumerate(modified) if m.get("role") == "user"),
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
            new_content = skill_block.lstrip("\n") + "\n\n" + modified[last_user_idx]["content"]
        else:
            # Append: skills stay at the recency-favoured end of the context,
            # immediately before generation on continuation turns.
            new_content = modified[last_user_idx]["content"] + skill_block

        modified[last_user_idx] = {"role": "user", "content": new_content}
        return self.agent.inference(modified)

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

    @classmethod
    def _is_dbbench_prompt(cls, content: str) -> bool:
        if not isinstance(content, str):
            return False
        text = content.lower()
        return (
            "help me operate a mysql database with sql" in text
            or ("action: operation" in text and "final answer:" in text and "mysql" in text)
        )

    @staticmethod
    def _dbbench_protocol() -> str:
        return (
            "---\n"
            "**SQL protocol reminder:** output exactly one valid action each turn.\n"
            "- To query or modify the database: Action: Operation + SQL block\n"
            "- When done: Action: Answer + Final Answer: [\"...\"]\n"
            "Never output placeholders. Never omit the Action line.\n"
            "For INSERT/UPDATE tasks, verify with a targeted SELECT before answering."
        )
