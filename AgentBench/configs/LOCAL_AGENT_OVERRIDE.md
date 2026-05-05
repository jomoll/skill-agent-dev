# Local Agent Configuration Override

After pulling updates, replace the `agent:` block in ALL config files with this:

```yaml
# Agent using local vLLM endpoint (LiteLLM proxy, OpenAI-compatible)
agent:
  module: src.client.agents.HTTPAgent
  parameters:
    url: "http://10.32.16.43:4000/chat/completions"
    headers:
      Authorization: "Bearer sk--jYBY-YIx5jNQqR0LPgqdQ"
      Content-Type: "application/json"
    body:
      model: "gpt-oss-120b"
      temperature: 0.0
      max_tokens: 8192
    prompter:
      name: role_content_dict
      args:
        agent_role: "assistant"   # OpenAI expects "assistant", not "agent"
```

## Affected files

All `skill_cycle_*.yaml` and `start_skill_task_*.yaml` and `start_task.yaml` in this directory.

## Quick apply (run from AgentBench root)

```bash
for f in configs/skill_cycle_*.yaml configs/start_skill_task_*.yaml configs/start_task.yaml; do
  echo "Check: $f"
done
```
