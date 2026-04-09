---
name: skeleton
description: Template and quality standard for learned skills — read-only
tags: []
---

# Skill Title

Use a title that names a reusable question-asking strategy, reasoning pattern,
or exploration approach for lateral thinking puzzles.
Prefer titles like `Systematic Witness Elimination` or
`Hidden Sensory Impairment Detection` over titles that reference only one
specific puzzle.

## Pattern Description

Write 1–3 paragraphs. State the central reusable lesson. Describe when this
question-asking or reasoning pattern improves puzzle-solving.

- Keep the opening general and reusable across different puzzles.
- Focus on a stable strategy, not one isolated benchmark case.

## When to Use This Skill

Bullet list of trigger conditions where this skill applies.

- Example: "When the story involves an unexpected emotional reaction, probe for
  hidden medical conditions or sensory impairments before assuming other motives"
- Example: "When a character's behavior seems irrational, ask about their prior
  knowledge of the situation"
- Example: "When early yes/no answers confirm an unusual setting, systematically
  lock in who/what/where before asking why"

## Common Failure Patterns

- Asking compound questions ("Was he blind AND exhausted?") — the host can only
  answer one clear yes/no at a time
- Repeating questions already answered or closely paraphrased (wastes rounds)
- Jumping to a final conclusion before confirming all answer-key sub-facts
- Asking vague questions like "Is there something unusual?" that produce no
  new information

## Recommended Patterns

**Pattern 1: structured broad-to-narrow elimination**
Start broad (confirm setting, number of people, time of day), then narrow
progressively to cause and motive. Each round should halve the remaining
uncertainty.

**Pattern 2: confirm implied facts before concluding**
Before guessing the answer, verify each implied element separately.
E.g. if blindness seems relevant, ask: "Did the character have a visual
impairment?" before drawing any conclusion.

**Pattern 3: target answer-key anchors directly**
The host judges progress against specific answer-key points. Prioritize
questions that directly target the key narrative elements: health, relationships,
prior events, objects, and sequences.

## Example Application

**Task:** "A man ate a bowl of soup at a restaurant and then killed himself..."

**Step-by-step:**
1. Confirm the immediate trigger: "Was eating the soup the direct cause of his
   distress?" → Yes
2. Probe prior context: "Had he eaten this type of soup before?" → Yes
3. Check for deception: "Was the soup different from what he expected?" → Yes
4. Verify the prior memory: "Did the soup remind him of a traumatic experience?" → Yes
5. Confirm the consequence: "Did realising the truth make him want to die?" → Yes
6. All answer-key points confirmed — state the conclusion.

## Success Indicators

- `finished=True` returned by the task
- All answer-key points discovered within 20 rounds (leaving safety margin)

## Failure Indicators

- Reaching round 25 without resolving key points (`finished=False`)
- `progress < 0.5` — fewer than half of answer-key clues confirmed
- Asking the same question rephrased across multiple rounds

---

A useful skill is 30–60 lines. Include example question sequences —
vague guidance does not reliably change agent behaviour.
