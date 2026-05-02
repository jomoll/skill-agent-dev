---
name: mutation_execute_then_stop
description: After a valid INSERT or UPDATE executes, stop issuing extra SQL and end with a minimal final answer.
tags:
- sql
- mutation
- protocol
version: 1
---

# Summary
Many mutation failures come from post-mutation behavior, not the mutation itself: extra unsupported SQL, malformed follow-up, or drifting into prose.

## Trigger
A mutation task has just executed without a SQL syntax or runtime error.

## Failure Pattern
The agent performs the correct `INSERT` or `UPDATE`, then:
- adds extra exploratory SQL,
- tries unsupported functions,
- writes long natural-language confirmation,
- or otherwise corrupts a potentially correct trajectory.

## Action Rule
After one successful mutation statement, immediately end the interaction with:
`Action: Answer`
`Final Answer: ["done"]`

Use additional SQL only if the mutation itself clearly failed.

## Verification Rule
Treat "no SQL error returned" as enough to stop for mutation tasks in this benchmark setting. Do not invent DB-specific verification functions or extra bookkeeping.

## Do Not
- Do not call functions like `MD5(...)`.
- Do not summarize the change in prose.
- Do not run a second SQL statement unless the first one failed.

## Example Pattern
Wrong:
```sql
UPDATE `Actress Filmography` SET `Year` = '2010', `Notes` = 'romantic comedy' WHERE `Title` = 'Jumping the Broom';
SELECT * FROM `Actress Filmography` WHERE `Title` = 'Jumping the Broom';
```

Correct:
```sql
UPDATE `Actress Filmography` SET `Year` = '2010', `Notes` = 'romantic comedy' WHERE `Title` = 'Jumping the Broom';
```

Then:
```text
Action: Answer
Final Answer: ["done"]
```
