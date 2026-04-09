---
name: skeleton
description: Template and quality standard for learned skills — read-only
tags: []
---

# Skill Title

Use a title that names a reusable strategic pattern, decision heuristic,
or resource-management rule for the card game.
Prefer titles like `Tempo Prioritization in Early Rounds` or
`Fish Trade Efficiency` over titles that reference only one specific game state.

## Pattern Description

Write 1–3 paragraphs. State the central reusable lesson. Describe when this
strategic pattern changes agent decisions and improves win rate.

- Keep the opening general and applicable across multiple game states.
- Focus on a stable pattern, not a one-off observation from a single game.

## When to Use This Skill

Bullet list of trigger conditions where this skill applies.

- Example: "When you have more cards in hand than the opponent, prioritise
  playing high-value fish to extend the tempo advantage"
- Example: "When HP is above 1200, focus on dealing damage; when below 800,
  switch to defensive plays"
- Example: "Against baseline2 opponents, early aggressive plays tend to be
  more effective than resource accumulation"

## Common Failure Patterns

- Playing cards with low damage output when a direct kill is available
- Ignoring the opponent's threat when their HP is already low
- Over-committing resources in the early game before board state is clear
- Making suboptimal guesses without reasoning through remaining card possibilities

## Recommended Patterns

**Pattern 1: threat assessment first**
Before choosing an action, evaluate: (a) can the opponent kill me this turn?
(b) can I kill the opponent this turn? Always handle (a) before (b).

**Pattern 2: card efficiency counting**
Track remaining cards and estimate opponent's likely hand. High-value plays
only make sense when the opponent cannot counter immediately.

**Pattern 3: baseline-specific adaptation**
Against baseline1 (aggressive): prioritise early board control.
Against baseline2 (reactive): exploit tempo by playing sub-optimally on
early turns to bait defensive responses, then strike.

## Example Application

**State:** Round 3, agent HP=1400, opponent HP=1100, agent has [card_A, card_B, card_C].

**Step-by-step:**
1. Check if opponent can deal 1400 damage this round → No (max damage ~600)
2. Check if agent can deal 1100 damage this round → Yes with card_A + card_B
3. Play card_A (600 damage), then card_B (600 damage) → opponent HP = -100 → win

## Success Indicators

- `win_round = 1` in the result metrics
- Winning within fewer rounds than the baseline agent typically requires

## Failure Indicators

- `win_round = 0` after using the skill
- Agent survives many rounds but cannot deal the final damage
- Agent takes excessive damage due to ignoring opponent threat assessment

---

A useful skill is 30–60 lines. Include concrete action sequences —
vague strategic advice does not reliably change agent behaviour.
