---
name: skeleton
description: Template and quality standard for learned skills — read-only
tags: []
---

# Skill Title

Use a title that names one reusable failure mechanism or decision rule.
Prefer titles like `Quote Reserved SQL Identifiers` or
`Verify Mutation With Targeted Select` over broad titles like
`SQL Best Practices`.

The skill must be behavior-changing, not just advisory.
It should tell the agent exactly when to activate the skill, what to do next,
and how to check whether it worked.

## Summary

Write 1 short paragraph.

- Name the single failure mechanism this skill addresses.
- State the concrete behavior change the skill should cause.
- Keep it reusable; do not encode benchmark-specific answers or hidden facts.

## Trigger

Bullet list of observable conditions that should activate the skill.
Triggers must be concrete and local to the interaction.

- Good: "A SQL error mentions syntax near `Rank` or `Order`"
- Good: "An `UPDATE` returns `[]` and the changed row has not been re-queried yet"
- Bad: "When the task is hard"

## Failure Pattern

Bullet list of the exact mistakes this skill prevents.
Reference actual actions, SQL fragments, output fields, or error messages.

- Good: "Using `UPDATE table SET Rank = 19` without backticks around `Rank`"
- Good: "Answering after `INSERT` without a follow-up `SELECT`"
- Bad: "Making mistakes in SQL"

## Action Rule

Write step-by-step operational guidance.
This section must change the agent's next action, query, or parsing step.

**Rule 1: primary behavior**
State the exact action to take and in what order.

**Rule 2: fallback**
State what to do if the primary action fails or returns no useful result.

## Verification Rule

State how to confirm that the skill worked.

- What result should the agent expect?
- What follow-up query, parsing step, or check should it perform?
- What should it avoid concluding too early?

## Do Not

Bullet list of 2–5 specific behaviors the agent should avoid.

- Good: "Do not assume `[]` proves an `UPDATE` changed a row"
- Good: "Do not use double quotes for MySQL identifiers when backticks are required"
- Bad: "Do not make errors"

## Example Trajectory

Provide one wrong trajectory and one correct trajectory showing 2–3 turns each.
Use a realistic task instruction and realistic actions/observations.
The wrong trajectory must show the exact failure this skill prevents.
The correct trajectory must show the exact behaviour this skill produces.

**WRONG trajectory:**
Task: "[task instruction]"
Turn 1 — Think: [agent reasoning that leads to the failure]
          Act: [wrong action]
Obs: [observation]
Turn 2 — Think: [reasoning showing the failure taking hold]
          Act: [wrong answer or wrong follow-up]
→ FAIL

**CORRECT trajectory:**
Task: "[same task instruction]"
Turn 1 — Think: [agent reasoning that applies this skill]
          Act: [correct first action]
Obs: [observation]
Turn 2 — Think: [agent reasoning leading to correct conclusion]
          Act: [correct answer or correct follow-up]
→ PASS

## Notes

Optional short bullets only if they improve generalization.

- Mention nearby edge cases only if they are part of the same mechanism.
- Do not add broad background explanations or mini-tutorials.

---

Quality bar:

- One skill = one failure mechanism.
- The skill must be specific enough that you can predict a different next action.
- Do not write style-only skills unless the dominant failure is invalid protocol.
- Do not restate an existing learned skill with synonyms.
- Prefer 25–50 lines of dense operational guidance over long generic prose.
