"""
SkillAwareAgent — wraps any AgentClient and injects the current skill library
into the first user message before each inference call.

Skills are always injected as structured JSON fields:

    {
      "selected_skills": ["skill1", "skill2"],
      "skill_documentation": {
        "skill1": "# description\\n\\ncontent...",
        "skill2": "content..."
      },
      "task": "<original prompt text or original JSON object>"
    }

If the first message is already a JSON object (e.g. MedAgentBench-style), skills
are merged into the existing object's top-level keys (preserving all other fields).
If the first message is plain text (OS Interaction, DBBench), it is wrapped in the
envelope above under the "task" key.

For DBBench-style SQL tasks a "protocol_reminder" field is also added to the
envelope, reinforcing the exact required output protocol.

Skills named "skeleton" (the read-only base template) are never injected.
If no skills exist and the task is not DBBench, history is passed through unchanged.
"""

import json
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

        modified = list(history)

        # Build or extend a JSON envelope for the first message.
        # Field order: skills first, then task, then protocol_reminder last.
        # Skills-first primes the model with behavioral constraints before it reads
        # the task; task+protocol_reminder last maximizes recency for the objective
        # and output format.
        try:
            existing = json.loads(first_content)
        except (json.JSONDecodeError, TypeError, AttributeError):
            existing = {"task": first_content}

        prompt_data: dict = {}

        if skills:
            prompt_data["selected_skills"] = [s["name"] for s in skills]
            prompt_data["skill_documentation"] = {
                s["name"]: (f"# {s['description']}\n\n" if s["description"] else "") + s["content"]
                for s in skills
            }

        # Merge all original fields (task, api, response_format, etc.) after skills
        prompt_data.update(existing)

        if is_dbbench:
            prompt_data["protocol_reminder"] = self._dbbench_protocol()

        modified[0] = {"role": "user", "content": json.dumps(prompt_data, indent=2)}
        return self.agent.inference(modified)

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
            "You are in a strict SQL benchmark. "
            "Output exactly one valid action each turn.\n"
            "- To query or modify the database: Action: Operation + SQL block\n"
            "- When done: Action: Answer + Final Answer: [\"...\"]\n"
            "Never output placeholders like \"normal\". Never omit the Action line.\n"
            "For INSERT/UPDATE tasks, verify with a targeted SELECT before answering."
        )
