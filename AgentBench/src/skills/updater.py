"""
SkillUpdater — proposes, validates, and applies learned skill edits.

ADD is blocked when the library is at capacity unless a REMOVE in the same batch
frees up a slot.

The updater makes a single inference call per proposal (no multi-agent pipeline).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.client.agent import AgentClient
from src.skills.repository import SkillRepository


_ALLOWED_ACTIONS = {"ADD", "MODIFY", "REMOVE"}
_READ_ONLY_BASE_SKILLS = {"skeleton"}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "unnamed_skill"


def _extract_balanced_json_block(text: str, open_char: str, close_char: str) -> Optional[str]:
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_fenced_payload(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text or "", re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _format_skill_summary(skill: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "tags": skill.get("tags") or [],
        "version": skill.get("version", 0),
    }


def _format_skill_with_stats(
    skill: Dict[str, Any],
    effectiveness: Optional[Dict[str, Any]],
) -> str:
    """One-line summary of a learned skill including provenance and runtime stats."""
    name = skill.get("name", "")
    desc = skill.get("description", "")
    version = skill.get("version", 1)
    prov = skill.get("provenance") or {}

    parts = [f"{name} (v{version})"]
    if desc:
        parts.append(f'"{desc}"')

    if prov:
        epoch = prov.get("epoch", "?")
        uc = prov.get("update_cycle", "?")
        ps = prov.get("probe_score", 0)
        pf = prov.get("fixes", 0)
        pr = prov.get("regressions", 0)
        parts.append(f"born=E{epoch}/UC{uc} probe={ps:+d}({pf}fix,{pr}regr)")

    eff = (effectiveness or {}).get(name)
    if eff:
        runs = eff.get("runs", 0)
        ef = eff.get("fixes", 0)
        er = eff.get("regressions", 0)
        parts.append(f"recent={ef}fix,{er}regr/{runs}runs")
    elif prov:
        parts.append("recent=no_data")

    return "  " + " | ".join(parts)


def _format_log(entries: List[Dict], prev_results: Optional[Dict[str, bool]] = None) -> str:
    lines: List[str] = []
    for entry in entries:
        sample_id = entry.get("sample_id", "unknown")
        status = entry.get("status", "unknown")
        is_correct = entry.get("is_correct", False)
        instruction = entry.get("instruction", "")
        query_type = entry.get("query_type", "other")
        error = entry.get("error")
        failure_tags = entry.get("failure_tags") or []
        actions = entry.get("agent_actions") or []
        history = entry.get("history") or []
        skill_names = [s.get("name", "") for s in entry.get("skill_snapshot_before", [])]

        transition = ""
        if prev_results is not None and sample_id in prev_results:
            prev = prev_results[sample_id]
            if prev and not is_correct:
                transition = " regression_from_prev_epoch=true"
            elif not prev and is_correct:
                transition = " recovery_from_prev_epoch=true"

        lines.append(
            f"[sample_id={sample_id} status={status} correct={is_correct}{transition}]"
        )
        if instruction:
            lines.append(f"Instruction: {instruction}")
        lines.append(f"Query type: {query_type}")
        if failure_tags:
            lines.append(f"Failure tags: {', '.join(failure_tags)}")
        if skill_names:
            lines.append(f"Learned skills before run: {', '.join(skill_names)}")
        if actions:
            lines.append("Agent actions:")
            for action in actions[:8]:
                lines.append(f"- {action}")
            if len(actions) > 8:
                lines.append(f"- ... ({len(actions) - 8} more)")
        if history:
            lines.append("Selected trace context:")
            for msg in history[-8:]:
                role = msg.get("role", "unknown")
                content = str(msg.get("content", "") or "").strip()
                if not content:
                    continue
                compact = re.sub(r"\s+", " ", content)
                if len(compact) > 400:
                    compact = compact[:400] + "..."
                lines.append(f"- {role}: {compact}")
        if error:
            lines.append(f"Error: {error}")
        lines.append("")
    return "\n".join(lines).strip()


def _infer_task_family(entries: List[Dict]) -> str:
    for entry in entries:
        instruction = str(entry.get("instruction", "") or "").lower()
        history = entry.get("history") or []
        history_text = " ".join(str(msg.get("content", "") or "") for msg in history).lower()
        if "mysql" in instruction or "mysql" in history_text or "action: operation" in history_text:
            return "DBBench SQL"
    return "AgentBench"


def _extract_json_array(text: str) -> List[Any]:
    """Extract a JSON array from model output with balanced parsing."""
    candidates = []
    fenced = _extract_fenced_payload(text or "")
    if fenced:
        candidates.append(fenced)
    candidates.append(text or "")

    for candidate in candidates:
        block = _extract_balanced_json_block(candidate.strip(), "[", "]")
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
    return []


def _build_prompt(
    entries: List[Dict],
    skill_repo: SkillRepository,
    max_proposals: int,
    max_learned_skills: int,
    prev_results: Optional[Dict[str, bool]] = None,
    skill_effectiveness: Optional[Dict[str, Any]] = None,
) -> str:
    all_skills = skill_repo.load_all()
    learned_skills = [s for s in all_skills if skill_repo.exists_in_learned(s["name"])]
    reference_skills = [s for s in all_skills if not skill_repo.exists_in_learned(s["name"])]

    skeleton = next((s for s in reference_skills if s["name"] == "skeleton"), None)
    non_skeleton_refs = [s for s in reference_skills if s["name"] != "skeleton"]

    editable_names = [s["name"] for s in learned_skills]
    if learned_skills:
        editable_skill_section = "\n".join(
            _format_skill_with_stats(s, skill_effectiveness) for s in learned_skills
        )
    else:
        editable_skill_section = "(none yet)"
    reference_skill_section = (
        json.dumps([_format_skill_summary(s) for s in non_skeleton_refs], indent=2, ensure_ascii=False)
        if non_skeleton_refs else "(none)"
    )
    skeleton_section = skeleton["content"] if skeleton else "(not found)"
    log_section = _format_log(entries, prev_results=prev_results)
    task_family = _infer_task_family(entries)

    slots_free = max_learned_skills - len(learned_skills)
    if slots_free == 0:
        skill_stats = (
            f"Learned skills in library: {len(learned_skills)} / {max_learned_skills} — LIBRARY FULL.\n"
            f"ADD is blocked. You MUST propose MODIFY or REMOVE (or both).\n"
            f"Skills with recent=0fix or recent=no_data are the weakest removal candidates.\n"
            f"To introduce a new skill, pair a REMOVE of the weakest existing skill with an ADD of the replacement in the same proposal array.\n"
            f"Editable learned skill names: {', '.join(editable_names)}"
        )
    elif slots_free <= 2:
        skill_stats = (
            f"Learned skills in library: {len(learned_skills)} / {max_learned_skills} — only {slots_free} slot(s) remaining.\n"
            f"Prefer REMOVE of a skill with recent=0fix,0regr (stale) + ADD of a better replacement over a plain ADD.\n"
            f"Editable learned skill names: {', '.join(editable_names) if editable_names else '(none yet)'}"
        )
    else:
        skill_stats = (
            f"Learned skills in library: {len(learned_skills)} / {max_learned_skills}\n"
            f"Editable learned skill names: {', '.join(editable_names) if editable_names else '(none yet)'}"
        )

    # When the library is full or nearly full the prompt asks for a REMOVE+ADD pair,
    # so allow 2 proposals per call in those cases regardless of the base max_proposals.
    effective_max_proposals = max(max_proposals, 2) if slots_free <= 2 else max_proposals

    return f"""You are helping improve an {task_family} agent's learned skill library.

You are the Skill Author. Read the current batch log and propose at most {effective_max_proposals}
skill edits as valid JSON.

--- SKILL CONTENT TEMPLATE ---
Every skill you write must follow this structure exactly. Each section is required.
{skeleton_section}
--- END TEMPLATE ---

Read-only reference/base skills for guidance only (never MODIFY or REMOVE these):
{reference_skill_section}

Editable learned skills:
{editable_skill_section}

{skill_stats}

--- PERFORMANCE LOG ---
{log_section}

Return ONLY a JSON array of proposed edits:
[
  {{
    "action": "ADD | MODIFY | REMOVE",
    "name": "snake_case_skill_name",
    "description": "one-line description",
    "content": "markdown body for the skill",
    "tags": ["optional", "tags"]
  }}
]

Rules:
- Focus on the CURRENT BATCH only.
- Use ADD when the batch reveals a distinct failure mechanism not covered by any existing learned skill.
- Use MODIFY when an existing skill covers the right mechanism but has a missing trigger, wrong example, or incomplete action rule — fix the specific gap, do not rewrite the whole skill.
- Use REMOVE when an existing skill is redundant (fully covered by another), too vague to change behavior, or is causing regressions visible in the log. A REMOVE + ADD pair is the correct way to replace a weak skill with a better one.
- Never MODIFY or REMOVE read-only base skills such as "skeleton".
- Prefer reusable capability skills over narrow one-task recipes, but do not broaden a skill so much that it stops changing behavior.
- One skill must target exactly one failure mechanism.
- A good skill must change the agent's next action, query, parsing step, or verification behavior.
- If a proposal would only change wording, tone, or answer style, reject it unless the dominant failure is invalid protocol.
- Keep concrete operational detail. Do not broaden into vague generic skills or mini-tutorials.
- Use specific commands, flags, SQL fragments, error messages, output patterns, or observable triggers as examples when possible.
- Use the selected trace context to identify whether the real issue is schema inference, row selection, column selection, identifier quoting, mutation construction, output parsing, answer formatting, or verification logic.
- Treat failure tags as strong hints about the dominant mechanism.
- For DBBench SQL tasks, prioritize protocol compliance, identifier quoting, schema exploration, mutation verification, and avoiding premature "cannot answer" responses.
- Do not propose generic "verify more", "format better", or "be careful" skills if an existing learned skill already covers that behavior.
- Do not restate an existing skill with synonyms. If the mechanism is already covered, return [] or propose a narrow MODIFY with a clear missing trigger or action rule.
- Do not encode benchmark-specific answers, row values, or hidden facts. Generalize the mechanism without leaking task content.
- Each proposal must be justified by at least one visible trigger in the trace, such as:
  - a specific SQL error
  - a missing or ambiguous schema field
  - an empty result after a mutation
  - a wrong comparison due to string/numeric confusion
  - a premature "cannot answer" conclusion
- Before proposing a skill, silently ask:
  1. What exact trigger activates this skill?
  2. What exact behavior changes because of it?
  3. Why would that flip at least one failing sample in this batch?
  4. Is this already covered by an existing learned skill?
- Skills should prefer realistic example identifiers from the trace pattern over placeholders like `table_name` and `column1`, but must remain mechanism-level and reusable.
- If there is not enough evidence for a good edit, return [].
""".strip()


class SkillUpdater:
    def __init__(
        self,
        agent: AgentClient,
        max_proposals: int = 5,
        max_learned_skills: int = 20,
    ) -> None:
        self.agent = agent
        self.max_proposals = max_proposals
        self.max_learned_skills = max_learned_skills

    def propose(
        self,
        entries: List[Dict],
        skill_repo: SkillRepository,
        prev_results: Optional[Dict[str, bool]] = None,
        skill_effectiveness: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Call the LLM and return raw (unvalidated) proposals."""
        prompt = _build_prompt(
            entries,
            skill_repo,
            self.max_proposals,
            self.max_learned_skills,
            prev_results=prev_results,
            skill_effectiveness=skill_effectiveness,
        )
        history = [{"role": "user", "content": prompt}]
        try:
            response = self.agent.inference(history)
            proposals = _extract_json_array(response)
            if not isinstance(proposals, list):
                print(f"[SkillUpdater] unexpected response type: {type(proposals)}")
                return []
            return [p for p in proposals if isinstance(p, dict)]
        except Exception as e:
            print(f"[SkillUpdater] inference failed: {e}")
            return []

    def validate(self, proposals: List[Dict], skill_repo: SkillRepository) -> List[Dict]:
        """Validate and normalize raw proposals before any evaluation/apply step."""
        valid: List[Dict] = []
        learned_names = {s["name"] for s in skill_repo.snapshot()}
        pending_remove_names = {
            _slugify(str(p.get("name", "")))
            for p in proposals
            if isinstance(p, dict) and str(p.get("action", "")).upper().strip() == "REMOVE"
        }

        add_slots_available = (
            self.max_learned_skills - skill_repo.learned_count() + len(pending_remove_names)
        )
        adds_reserved = 0

        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue

            action = str(proposal.get("action", "")).upper().strip()
            if action not in _ALLOWED_ACTIONS:
                continue

            name = _slugify(str(proposal.get("name", "") or ""))
            description = str(proposal.get("description", "") or "").strip()
            content = str(proposal.get("content", "") or "").strip()
            tags = proposal.get("tags") or []

            if name in _READ_ONLY_BASE_SKILLS:
                print("[SkillUpdater] attempt to modify/remove base skeleton rejected")
                continue

            normalized = {
                "action": action,
                "name": name,
                "description": description,
                "content": content,
                "tags": tags,
            }

            if action == "ADD":
                if name in learned_names:
                    print(f"[SkillUpdater] ADD for existing learned skill rejected: {name}")
                    continue
                if not content:
                    continue
                if adds_reserved >= add_slots_available:
                    print(f"[SkillUpdater] ADD blocked at capacity: {name}")
                    continue
                adds_reserved += 1
                print(f"[SkillUpdater] ADD skill: {name}")
                valid.append(normalized)
                continue

            if action == "MODIFY":
                if name not in learned_names:
                    print(f"[SkillUpdater] MODIFY for unknown learned skill rejected: {name}")
                    continue
                if not content:
                    continue
                print(f"[SkillUpdater] MODIFY skill: {name}")
                valid.append(normalized)
                continue

            if action == "REMOVE":
                if name not in learned_names:
                    print(f"[SkillUpdater] REMOVE for unknown learned skill rejected: {name}")
                    continue
                print(f"[SkillUpdater] REMOVE skill: {name}")
                valid.append(normalized)

        return valid

    def apply(self, proposals: List[Dict], skill_repo: SkillRepository) -> List[Dict]:
        applied: List[Dict] = []
        for proposal in proposals:
            action = proposal["action"]
            name = proposal["name"]
            description = proposal.get("description", "")
            content = proposal.get("content", "")
            tags = proposal.get("tags") or []
            provenance = proposal.get("_provenance")  # attached by cycle, not from LLM

            if action == "ADD":
                skill_repo.add(name, description, content, tags=tags, provenance=provenance)
            elif action == "MODIFY":
                skill_repo.modify(name, description, content, tags=tags, provenance=provenance)
            elif action == "REMOVE":
                skill_repo.delete(name)
            else:
                continue
            # Strip internal keys before logging
            applied.append({k: v for k, v in proposal.items() if not k.startswith("_")})
        return applied
