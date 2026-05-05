"""
MemoryUpdater — generates per-sample correction notes from failing agent traces
and manages the memory file (memory.json).

Prompt design matches the original MedAgentBench-v2 paper (Appendix A.2):
  - One LLM call per failing sample
  - Includes task description, agent response, eval output (ref_sol), and current memory
  - Output is plain prose starting with "when asked ..."
  - No JSON array; bullets are appended as plain strings

The memory is a flat JSON list of strings. After each update cycle:
  1. propose() calls the LLM once per failing entry.
  2. New bullets are appended to the list.
  3. condense() is called when the list reaches max_bullets capacity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _render_memory_block(bullets: List[str]) -> str:
    if not bullets:
        return "<memory>\n</memory>"
    lines = ["<memory>", "Correction notes from past experience:"]
    lines += [f"- {b}" for b in bullets]
    lines.append("</memory>")
    return "\n".join(lines)


def _format_agent_response(entry: Dict) -> str:
    task_result = entry.get("task_result")
    if task_result is not None:
        return str(task_result)[:1000]
    history = entry.get("history") or []
    for msg in reversed(history):
        if msg.get("role") in ("agent", "assistant"):
            content = str(msg.get("content", "") or "").strip()
            if content:
                return content[:1000]
    return "(no response captured)"


def _format_eval_output(entry: Dict) -> str:
    ground_truth = entry.get("ground_truth")
    is_correct = entry.get("is_correct", False)
    parts = []
    if ground_truth is not None:
        parts.append(f"ref_sol: {ground_truth}")
    parts.append(str(is_correct))
    return "\n".join(parts)


class MemoryUpdater:
    def __init__(self, agent, max_bullets: int = 20) -> None:
        self.agent = agent
        self.max_bullets = max_bullets

    def propose_one(self, entry: Dict, current_bullets: List[str]) -> Optional[str]:
        """Generate a single memory note for one failing sample (paper-style prompt)."""
        instruction = str(entry.get("instruction", "") or entry.get("sample_id", ""))
        context = str(entry.get("context", "") or "")
        task_descr = f"Instruction:\n{instruction}"
        if context:
            task_descr += f"\nContext:\n{context}"

        agent_response = _format_agent_response(entry)
        eval_output = _format_eval_output(entry)
        current_prompt = _render_memory_block(current_bullets)

        prompt = (
            "Add memory to the current_prompt. Since the current agent doesn't handle this task "
            "correctly, write instructions for a correct approach to the agent's memory so when it "
            "sees the task again, it gets it right. Think about the task description, the agent's "
            "previous response, and what the evaluation function tests to figure out why the agent "
            "got the wrong response. Use 1-3 sentences to correct its MAIN mistake. "
            "Start with \"when asked...\"\n\n"
            "Example Response: when asked \"If low, then order replacement IV magnesium according "
            "to dosing instructions.\", low indicates a value below 1.5 mg/dL.\n\n"
            f"<task_description>\n{task_descr}\n</task_description>\n\n"
            f"<agent_response>\n{agent_response}\n</agent_response>\n\n"
            f"<eval_output>\n{eval_output}\n</eval_output>\n\n"
            f"<current_prompt>\n{current_prompt}\n</current_prompt>"
        )

        try:
            response = self.agent.inference([{"role": "user", "content": prompt}])
            bullet = response.strip()
            if bullet:
                print(f"[MemoryUpdater] new note: {bullet[:120]}")
                return bullet
        except Exception as e:
            print(f"[MemoryUpdater] propose_one failed: {e}")
        return None

    def propose(self, failing_entries: List[Dict], current_bullets: List[str]) -> List[str]:
        """Call the LLM once per failing entry to generate correction notes."""
        if not failing_entries:
            return []
        new_bullets: List[str] = []
        for entry in failing_entries:
            bullet = self.propose_one(entry, current_bullets + new_bullets)
            if bullet:
                new_bullets.append(bullet)
        return new_bullets

    def condense(self, bullets: List[str]) -> List[str]:
        """Ask the LLM to condense the memory list when at capacity."""
        target = max(10, self.max_bullets // 2)
        bullets_text = "\n".join(f"- {b}" for b in bullets)
        prompt = (
            f"The following memory list has {len(bullets)} entries. "
            f"Please condense it to at most {target} entries by merging similar notes "
            "and keeping only the most impactful ones. Preserve the 'when asked...' format "
            "where applicable.\n\n"
            f"Current entries:\n{bullets_text}\n\n"
            f"Return ONLY a JSON array of at most {target} strings:\n"
            '["entry 1", "entry 2"]'
        )
        try:
            from typing import Any
            candidates = []
            text = self.agent.inference([{"role": "user", "content": prompt}])
            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text or "", re.DOTALL)
            if fenced:
                candidates.append(fenced.group(1).strip())
            candidates.append(text or "")
            for candidate in candidates:
                start = candidate.find("[")
                if start == -1:
                    continue
                try:
                    data = json.loads(candidate[start:])
                    if isinstance(data, list):
                        result = [str(b).strip() for b in data if isinstance(b, str) and b.strip()]
                        if result:
                            print(f"[MemoryUpdater] condensed {len(bullets)} → {len(result)} entries")
                            return result[:target]
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"[MemoryUpdater] condense failed: {e}")
        return bullets[:target]

    def update(self, memory_path: Path, failing_entries: List[Dict]) -> List[str]:
        """Load memory, propose new notes, append, save and return updated list."""
        current: List[str] = []
        if memory_path.exists():
            try:
                current = json.loads(memory_path.read_text(encoding="utf-8"))
            except Exception:
                current = []

        new_bullets = self.propose(failing_entries, current)
        updated = current + new_bullets

        memory_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return updated
