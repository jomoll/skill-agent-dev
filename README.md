# skill-agent-dev

Self-improving LLM agent framework using a GRPO-inspired skill cycle. Agents learn reusable behavioral skills from their own failures across multiple benchmarks.

```
emnlp26/
├── AgentBench/        # OS Interaction, DBBench, LTP, Card Game, ALFWorld skill cycles
├── MedAgentBench/     # FHIR medical records skill cycle
└── FHIR-AgentBench/   # Native FHIR-AgentBench skill cycle
```

## How it works

After each batch of task episodes, a skill-writing LLM observes the agent's failure traces and proposes additions, modifications, or removals to a markdown skill library. Candidates are scored on a balanced probe set (fixes − regressions), and the winner is applied. Skills are injected as structured JSON context into every subsequent inference call — no fine-tuning required.

---

## Setup

### Prerequisites

- Python 3.9 (recommended)
- [Docker](https://www.docker.com/) installed and running
- Google Cloud credentials for Vertex AI (`gcloud auth application-default login`)

### 1. AgentBench (OS Interaction + DBBench + LTP + Card Game + ALFWorld)

```bash
cd AgentBench
conda create -n agent-bench python=3.10
conda activate agent-bench
pip install -r requirements.txt
```

`requirements.txt` includes `agentrl-worker`, which provides the `agentrl.worker` module used by all task workers and the skill cycle.

Pull and build Docker images:

```bash
# OS Interaction
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu

# DBBench — pin to MySQL 8 (MySQL 9+ removed MD5 which the benchmark requires)
docker pull mysql:8

# LTP
docker pull longinyu/agentbench-ltp

# Card Game
docker pull longinyu/agentbench-card_game

# ALFWorld
docker pull longinyu/agentbench-alfworld
```

### 2. MedAgentBench

```bash
cd MedAgentBench
conda create -n medagentbench python=3.9
conda activate medagentbench
pip install -r requirements.txt
```

Pull and start the FHIR server:

```bash
docker pull jyxsu6/medagentbench:latest
docker tag jyxsu6/medagentbench:latest medagentbench
docker run -p 8080:8080 medagentbench
```

Wait until the console shows "Started Application in XXX seconds", then verify at `http://localhost:8080/`.

Download the reference solution file into `MedAgentBench/src/server/tasks/medagentbench/refsol.py` from the [Stanford Box link](https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho).

### 3. FHIR-AgentBench

```bash
cd FHIR-AgentBench
conda create -n fhir-agentbench python=3.11
conda activate fhir-agentbench
pip install -r requirements.txt
```

Create `FHIR-AgentBench/config.yml` as described in that repo's README. The skill cycle
uses the same Google Cloud Healthcare/FHIR credentials and LiteLLM model settings as
the original `run_agent.py` workflow.

---

## Running the skill cycle

All benchmarks use separate controller ports and can run in parallel.

Tasks that manage Docker containers internally (DBBench, OS Interaction) require a Redis instance for container allocation and a pre-created Docker network. Run these once before starting any task worker:

```bash
# Redis (keep running in background)
docker run --rm --name agentbench-redis -p 6379:6379 redis:7

# Docker networks (one-time, survives reboots with --driver bridge)
docker network create dbbench_default || true
docker network create os_interaction_default || true
```

> **Linux note**: The task worker requires Docker 20.10+ for `--add-host host.docker.internal:host-gateway` support. Verify with `docker --version`. Also ensure your user is in the `docker` group (`sudo usermod -aG docker $USER`, then re-login) so the task worker can manage containers without sudo.

| Benchmark | Controller port | Worker base port |
|---|---|---|
| OS Interaction | 5040 | 5041 |
| DBBench | 5010 | 5011 |
| LTP | 5020 | 5021 |
| Card Game | 5030 | 5031 |
| ALFWorld | 5060 | 5061 |
| MedAgentBench | 5001 (default) | 5002 |
| FHIR-AgentBench | none | none |

### OS Interaction

```bash
# Terminal 1 — start task worker
cd AgentBench && conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_os.yaml

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_os.yaml --run-name run_001
```

### DBBench

```bash
# Terminal 1 — start task worker (Redis and dbbench_default network must already be running)
cd AgentBench && conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_dbbench.yaml --controller-port 5010 --base-port 5011

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_dbbench.yaml --run-name run_001
```

### LTP (Lateral Thinking Puzzle)

The LTP task uses a second Gemini agent as the puzzle host (answers yes/no questions).
Host credentials are automatically mounted from `~/.config/gcloud` into the Docker container.

```bash
# Generate data splits (one-time)
cd AgentBench && python data/lateralthinkingpuzzle/split_dataset.py

# Terminal 1 — start task worker
cd AgentBench && conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_ltp.yaml --controller-port 5020 --base-port 5021

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_ltp.yaml --run-name run_001
```

### Card Game

The Card Game skill-cycle split is procedurally generated from `cg-std` with
`test_time=40`: 80 dev samples, 60 val samples, and 20 held-out test samples.
The current cycle config uses `update_every=40`, so each epoch has two serial
dev batches.

```bash
# Generate data splits (one-time)
cd AgentBench && python data/card_game/split_dataset.py

# Terminal 1 — start task worker
cd AgentBench && conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_card_game.yaml --controller-port 5030 --base-port 5031

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_card_game.yaml --run-name run_001
```

### ALFWorld

```bash
# Terminal 1 — start task worker
cd AgentBench && conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_alfworld.yaml --controller-port 5060 --base-port 5061

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_alfworld.yaml --run-name run_001
```

### MedAgentBench

```bash
# Terminal 1 — start FHIR server (if not already running)
docker run -p 8080:8080 medagentbench

# Terminal 2 — start task worker
cd MedAgentBench && conda activate medagentbench
python -m src.start_task -a --config configs/start_task.yaml

# Terminal 3 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle.yaml --run-name run_001
```

### FHIR-AgentBench

FHIR-AgentBench does not use the AgentBench task controller. Its skill cycle wraps
the existing FHIR agents directly, injects learned markdown skills into each agent's
system prompt, and scores samples with a cached per-sample answer judge.

```bash
cd FHIR-AgentBench && conda activate fhir-agentbench
python skill_cycle.py --config configs/skill_cycle.yaml --run-name run_001
```

If a run is interrupted or killed, resume it from the last completed samples:

```bash
python skill_cycle.py --config configs/skill_cycle.yaml --run-name run_001 --resume
```

The default config uses `multi_turn_code_resource`, `openai/gpt-oss-120b`, the CSV at
`final_dataset/questions_answers_sql_fhir.csv`, the CSV `train` split as dev (capped at 80 samples), and
the CSV `valid` split as val (capped at 40 samples), with skill updates running every 20 samples. For local vLLM/LiteLLM-compatible models, set
`agent.base_url`, `updater.base_url`, and `eval.base_url` in
`FHIR-AgentBench/configs/skill_cycle.yaml`.

FHIR-AgentBench uses the same grouped proposal-ranking shape as AgentBench and
MedAgentBench: `cycle.grpo_k` is the total number of proposal calls per update,
cycled over the largest failure modes, and each validated proposal is ranked on
the same probe set against current-skill baseline fixes/regressions before a
single winner is applied.

### Evaluating with a manual skill pack

To run a fixed set of skills against a split (useful as an upper-bound reference):

```bash
cd AgentBench
python -m src.run_manual_skills --config configs/manual_skills_dbbench.yaml --split val
```

---

## Data splits

| Benchmark | Dev | Val | Test | Split strategy |
|---|---|---|---|---|
| MedAgentBench | 126 | 84 | 90 | 60/40 within tasks 1–5,8,9; tasks 6/7/10 held out (OOD) |
| DBBench | 240 | 124 | 60 | 176 real (60/40 of standard.jsonl by query type) + 64 synthetic aggregation; dev.jsonl held out |
| OS Interaction | 79 | 56 | 35 | 60/40 of worlds 1–5,7 stratified per world; world 6 + dev.json held out |
| LTP | 30 | 20 | 20 | 60/40 of standard.xlsx; dev.xlsx held out (IDs offset by 50 to avoid collision) |
| Card Game | 80 | 60 | 20 | 20/15/5 reps × 4 combos; procedurally generated (`cg-std.test_time=40`) |
| ALFWorld | 26 | 24 | 20 | Stratified 60/40 of standard.json by task type; dev.json held out |
| FHIR-AgentBench | configurable | configurable | original CSV test split | Defaults to capped train/valid rows from `questions_answers_sql_fhir.csv` |

### DBBench synthetic extension

The four aggregation sub-types (SUM, MIN, MAX, AVG) originally had only 4 dev samples each, which is too few for reliable skill learning. `data/dbbench/generate_synthetic.py` generates 16 additional dev-only samples per type by deriving new questions from the same tables already in `standard.jsonl`, varying the aggregated column and WHERE conditions. Answers are verified with SQLite. Synthetic samples use IDs ≥ 10000 and are stored in `synthetic_dev.json`; `split_dataset.py` appends them to `split_dev.json` automatically.

### ALFWorld and MedAgentBench

ALFWorld scenarios live inside the Docker image — synthetic extension requires the running container and was not attempted offline. MedAgentBench tasks 2–10 require a live FHIR server for answer verification — only task 1 carries pre-verified `sol` fields, which is insufficient for meaningful extension without the server.

Regenerate splits:

```bash
python AgentBench/data/dbbench/generate_synthetic.py   # regenerate synthetic aggregation samples (run first)
python AgentBench/data/dbbench/split_dataset.py
python AgentBench/data/os_interaction/split_dataset.py
python AgentBench/data/lateralthinkingpuzzle/split_dataset.py
python AgentBench/data/card_game/split_dataset.py
python AgentBench/data/alfworld/split_dataset.py
python MedAgentBench/data/medagentbench/split_dataset.py
```

---

## Configuration

Key config files:

| File | Purpose |
|---|---|
| `AgentBench/configs/skill_cycle_os.yaml` | OS skill cycle hyperparameters |
| `AgentBench/configs/skill_cycle_dbbench.yaml` | DBBench skill cycle hyperparameters |
| `AgentBench/configs/skill_cycle_ltp.yaml` | LTP skill cycle hyperparameters |
| `AgentBench/configs/skill_cycle_card_game.yaml` | Card Game skill cycle hyperparameters |
| `AgentBench/configs/skill_cycle_alfworld.yaml` | ALFWorld skill cycle hyperparameters |
| `MedAgentBench/configs/skill_cycle.yaml` | MedAgentBench skill cycle hyperparameters |
| `FHIR-AgentBench/configs/skill_cycle.yaml` | FHIR-AgentBench native skill cycle hyperparameters |
| `AgentBench/configs/agents/gemini-chat.yaml` | Vertex AI Gemini agent config |
| `MedAgentBench/configs/agents/vertex-gemini.yaml` | Vertex AI Gemini agent config |

The agent used is Gemini Flash Lite via Vertex AI. Authentication uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) — run `gcloud auth application-default login` before starting.

---

## Skill library structure

Skills are stored as markdown files with YAML frontmatter:

```
AgentBench/skills/
├── os/base/            # read-only OS base skills
├── dbbench/base/       # read-only DBBench base skills
├── ltp/base/           # read-only LTP base skills
├── card_game/base/     # read-only Card Game base skills
├── alfworld/base/      # read-only ALFWorld base skills
└── base/               # shared skeleton template

MedAgentBench/skills/
└── base/               # read-only MedAgentBench base skills

FHIR-AgentBench/skills/
└── base/               # read-only FHIR-AgentBench skill template
```

Learned skills are written to `outputs/<run>/skills/learned/` during training and loaded fresh on every inference call.

---

## Framework modifications

Changes from the original upstream codebase, grouped by concern.

### 1. Failure taxonomy — mechanism-based classification
**File:** `AgentBench/src/skills/updater.py` (`SkillUpdater.classify_failures`)

- The trace passed to the classifier is the **full action sequence** (all turns), with each
  action truncated to 160 characters. The original implementation passed only the last 2
  actions; the failure mechanism is often in the middle of a trace, so the full trace is needed.
- Instead of a closed required vocabulary, the classifier prompt injects **example labels** that
  show the right granularity. Labels must be specific enough that two different labels imply two
  different skills (e.g. `sql_max_on_text_column`, not `wrong_method_for_goal`). The examples
  are illustrative only — the classifier is instructed to generate new specific labels freely
  and only reuse prior-epoch labels when the mechanism genuinely recurs.

### 2. Skill injection point — prefix on first decision, suffix on continuations
**File:** `AgentBench/src/client/agents/skill_aware_agent.py` (`SkillAwareAgent.inference`)

- **First agent decision** (no prior assistant/agent turn): skills are **prepended** to the
  last user message so the model reads them before the task instruction, interrupting
  reflexive first-action behaviour.
- **Continuation turns** (prior agent turn exists): skills are **appended** after the latest
  observation, keeping them at the recency-favoured end of context (previous behaviour).

### 3. Task-type classification in system prompts
**Files:**
- `AgentBench/src/server/tasks/os_interaction/task.py`
- `AgentBench/src/server/tasks/dbbench/__init__.py`

A paragraph appended to each task's system prompt requires the agent to classify the task
before its first action.  Both benchmarks use the same A/B/C scheme:

| Type | OS | DBBench |
|------|----|---------|
| **A** execute-and-report | run commands, report result | query/aggregate live data, report result |
| **B** generate-artifact / modify-and-verify | produce script as text without executing | mutate database, verify with SELECT |
| **C** static-knowledge | answer from general Linux knowledge | answer from task description alone |

**Exact text appended to the OS Interaction system prompt** (after the action descriptions):
```
Before issuing your first action, your Think step must classify the task into exactly one of these types:
- Type A (execute-and-report): the task asks for a concrete value obtainable only by running commands on the live system (counts, sizes, process states, file contents, etc.). Run the relevant commands and report the result.
- Type B (generate-artifact): the task asks you to produce a script, command, or other text artifact without executing it. Return the artifact directly in answer().
- Type C (static-knowledge): the task can be answered from general knowledge without touching the live system. Answer directly without running any command.

State the type at the start of your first Think step before doing anything else.
```

**Exact text appended to the DBBench system prompt** (after the existing instructions):
```
Before your first SQL action, classify the task into exactly one of these types and state it explicitly:
- Type A (execute-and-report): the answer requires querying or aggregating live data (SELECT, aggregate, compare, rank). Run the necessary SQL and report the result.
- Type B (modify-and-verify): the task requires changing the database (INSERT, UPDATE, or DELETE). Execute the mutation, then verify with a targeted SELECT before answering.
- Type C (static-knowledge): the answer can be derived from the task description alone without querying the database.
State the type and your intended first action at the start of your explanation before writing any SQL.
```

### 4. Skill template — Example Trajectory replaces Example Pattern
**Files:**
- `AgentBench/skills/base/skeleton.md`
- `AgentBench/src/skills/updater.py` (`_build_prompt` generation rules)

The `## Example Pattern` section (static wrong/correct code pair) is replaced by
`## Example Trajectory`, requiring one wrong and one correct 2–3 turn trajectory
(`Think → Act → Obs → Think → Act`).  The generation rules in `_build_prompt` are updated
to require trajectory examples and prohibit static code pairs.
