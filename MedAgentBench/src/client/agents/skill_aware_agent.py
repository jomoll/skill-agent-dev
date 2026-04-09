"""
SkillAwareAgent — wraps any AgentClient and injects the current skill library
into the first user message before each inference call.

When the first message is a JSON-structured task prompt (as produced by
_build_task_prompt in the task server), skills are injected into the dedicated
`selected_skills` and `skill_documentation` fields:

    {
      "phase": "task_execution",
      "task": { "description": "...", "context": "..." },
      "selected_skills": ["magnesium_threshold", "finish_format"],
      "skill_documentation": {
        "magnesium_threshold": "# Mg < 1.9 mEq/L requires IV replacement\n...",
        "finish_format": "# FINISH format\n..."
      },
      "api": { ... },
      "response_format": { ... }
    }

If the first message is plain text (legacy format), falls back to prepending
a [SKILLS] block before the task text.

Skills named "skeleton" (the read-only base template) are never injected.
If no learned skills exist the history is passed through unchanged.
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
        if not skills:
            return self.agent.inference(history)

        modified = list(history)
        first_content = modified[0]["content"]

        try:
            prompt_data = json.loads(first_content)
            prompt_data["selected_skills"] = [s["name"] for s in skills]
            prompt_data["skill_documentation"] = {
                s["name"]: (f"# {s['description']}\n\n" if s["description"] else "") + s["content"]
                for s in skills
            }
            modified[0] = {"role": "user", "content": json.dumps(prompt_data, indent=2)}
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Fallback for plain-text prompts
            skill_block = self._format_skills_text(skills)
            modified[0] = {"role": "user", "content": skill_block + first_content}

        return self.agent.inference(modified)

    @staticmethod
    def _format_skills_text(skills: List[dict]) -> str:
        lines = ["[SKILLS]\n"]
        for skill in skills:
            lines.append(f"--- skill: {skill['name']} ---\n")
            if skill["description"]:
                lines.append(f"# {skill['description']}\n")
            lines.append(skill["content"])
            lines.append("\n\n")
        lines.append("[END SKILLS]\n\n[TASK]\n")
        return "".join(lines)
