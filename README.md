# skill-agent-dev

Self-improving LLM agent framework using a GRPO-inspired skill cycle. Agents learn reusable behavioral skills from their own failures across multiple benchmarks.

```
skill-agent-dev/
├── AgentBench/          # OS Interaction, DBBench, LTP, Card Game, ALFWorld skill cycles
├── MedAgentBench/       # FHIR medical records skill cycle (original 10 task types, v1 data)
├── MedAgentBench-v2/    # FHIR skill cycle on new clinical tasks (10 redesigned task types)
└── FHIR-AgentBench/     # Native FHIR-AgentBench skill cycle
```

## How it works

After each batch of task episodes, a skill-writing LLM observes the agent's failure traces and proposes additions, modifications, or removals to a markdown skill library. Candidates are scored on a balanced probe set (fixes − regressions), and the winner is applied only if it improves net score without exceeding baseline regressions. Skills are injected as structured JSON context into every subsequent inference call — no fine-tuning required.

### Memory comparator

Each benchmark also ships a `memory_cycle` alongside the `skill_cycle`. The memory cycle uses the **same agent, same updater model, same epoch/batch schedule, and same probe/val evaluation** as the skill cycle. The only difference: instead of proposing validated skill files, the updater synthesizes 1–3 natural-language correction bullets after each batch and appends them to a `<memory>` block in the agent's system prompt. Memory is applied unconditionally (no acceptance gate), making it a clean "no structure, no gate" baseline against which the skill cycle's advantages can be isolated.

This design answers two questions simultaneously: (1) does structured, auditable skill encoding beat cheap natural-language memory, and (2) how much of the skill cycle's gain comes from the regression-budget acceptance gate vs. the representation itself?

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

### 3. MedAgentBench-v2

MedAgentBench-v2 uses the same FHIR Docker image as MedAgentBench — the server already contains Procedure, Condition, and all Observation codes required by the new task types. No additional data loading is needed.

The new task types (redesigned from the ground up vs. v1) are qualitatively harder: multi-step decision trees, time-window reasoning, coordinated writes across multiple FHIR resource types, and safety protocols. Measured baseline performance on these tasks is ~64% (vs. ~70% on v1 tasks), providing more headroom for skill learning.

Two agent tools are added relative to MedAgentBench: `fhir_procedure_search` and `fhir_condition_search`. These are structural requirements — several task types cannot be attempted without reading Procedure or Condition resources — not performance boosters.

MedAgentBench-v2 has identical dependencies to MedAgentBench — reuse the same conda environment:

```bash
conda activate medagentbench
```

Start the same FHIR server (shared with MedAgentBench, same port):

```bash
docker run -p 8080:8080 medagentbench
```

The reference solution (`new_refsol.py`) is bundled directly in `MedAgentBench-v2/src/server/tasks/medagentbench/` — no separate download required.

Generate data splits (one-time):

```bash
cd MedAgentBench-v2
python data/medagentbench/split_dataset.py
```

### 4. FHIR-AgentBench

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
| MedAgentBench-v2 | 5070 | 5071 |
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

### MedAgentBench-v2

```bash
# Terminal 1 — start FHIR server (shared image, same port as MedAgentBench)
docker run -p 8080:8080 medagentbench

# Terminal 2 — start task worker
cd MedAgentBench-v2 && conda activate medagentbench
python -m src.start_task -a --config configs/start_task.yaml --controller-port 5070 --base-port 5071

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

## Running the memory cycle

The memory cycle is a drop-in comparator that runs on the same agent and data as the skill cycle. Replace `skill_cycle` with `memory_cycle` in the commands above:

```bash
# AgentBench (any task type)
python -m src.memory_cycle --config configs/memory_cycle_os.yaml --run-name mem_001

# MedAgentBench
python -m src.memory_cycle --config configs/memory_cycle.yaml --run-name mem_001

# MedAgentBench-v2
python -m src.memory_cycle --config configs/memory_cycle.yaml --run-name mem_001

# FHIR-AgentBench
python memory_cycle.py --config configs/memory_cycle.yaml --run-name mem_001
```

The memory cycle shares all infrastructure with the skill cycle (task workers, ports, FHIR server). No additional setup is needed.

**Design principles for a fair comparison:**
- Same base agent configuration — no enhanced system prompt, no additional tools
- Same dev/val splits, same batch size, same update cadence (one update per batch)
- Same updater model and failure classification step
- Memory bullets are appended unconditionally (no probe-based acceptance gate) — this is intentional: the gate is one of the variables being tested
- Val score is tracked identically, enabling direct learning-curve comparison

**Implementation across benchmarks:**

| Benchmark | Memory agent | Memory storage | Bullet synthesis |
|---|---|---|---|
| AgentBench / MedAgentBench / MedAgentBench-v2 | `MemoryAwareAgent` wrapping base agent | `<memory>` block in system prompt | Same updater model as skills |
| FHIR-AgentBench | Memory block prepended to agent system prompt per run | In-memory list serialized to `memory.md` in run dir | Same updater |

Key config knobs (in `memory_cycle.yaml` for each benchmark):

```yaml
memory:
  max_bullets: 10          # evict oldest when exceeded
  bullets_per_update: 3    # max new bullets synthesized per batch
```

---

## Data splits

| Benchmark | Dev | Val | Test | Split strategy |
|---|---|---|---|---|
| MedAgentBench | 126 | 84 | 90 | 60/40 within tasks 1–5,8,9; tasks 6/7/10 held out (OOD) |
| MedAgentBench-v2 | 126 | 84 | 90 | 60/40 of tasks 1–4,6,8,9 (18/12 per type); tasks 5,7,10 held out (OOD) |
| DBBench | 240 | 124 | 60 | 176 real (60/40 of standard.jsonl by query type) + 64 synthetic aggregation; dev.jsonl held out |
| OS Interaction | 79 | 56 | 35 | 60/40 of worlds 1–5,7 stratified per world; world 6 + dev.json held out |
| LTP | 30 | 20 | 20 | 60/40 of standard.xlsx; dev.xlsx held out (IDs offset by 50 to avoid collision) |
| Card Game | 80 | 60 | 20 | 20/15/5 reps × 4 combos; procedurally generated (`cg-std.test_time=40`) |
| ALFWorld | 26 | 24 | 20 | Stratified 60/40 of standard.json by task type; dev.json held out |
| FHIR-AgentBench | configurable | configurable | original CSV test split | Defaults to capped train/valid rows from `questions_answers_sql_fhir.csv` |

### MedAgentBench-v2 task types

The ten task categories in `new_patient_tasks.json` are completely redesigned from v1:

| Task | Clinical workflow | FHIR resources |
|---|---|---|
| 1 | CT Abd/Pelvis surveillance — order if >12 months old | Procedure (read), ServiceRequest (write) |
| 2 | DVT prophylaxis reconciliation — ensure exactly one heparin order | MedicationRequest (read + write) |
| 3 | Average heart rate over 6h and 12h windows | Observation (read only) |
| 4 | Urinary catheter dwell check — remove order if >48 hours | Procedure + ServiceRequest (read + write) |
| 5 | Renal mass protocol — CT + IR referral if diagnosis present and CT stale | Condition + Procedure + ServiceRequest (read + write) |
| 6 | Thyroid protocol — levothyroxine or repeat labs based on TSH/FT4 branching | Observation (read), MedicationRequest + ServiceRequest (write) |
| 7 | QTc safety — ECG order + discontinue QT-prolonging drug if QTc >500 ms | Observation + MedicationRequest (read + write) |
| 8 | Naloxone coverage — add naloxone if active opioid without naloxone | MedicationRequest (read + write) |
| 9 | Influenza vaccine recall — order if last shot >365 days ago | Procedure (read), ServiceRequest (write) |
| 10 | COVID-19 booster — order if last vaccine >12 months ago | Procedure + MedicationRequest (read + write) |

All evaluations apply a `meta.lastUpdated ≤ 2025-01-01` cutoff on baseline reads to prevent evaluator data leakage from agent writes during test execution.

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
python MedAgentBench-v2/data/medagentbench/split_dataset.py
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
| `AgentBench/configs/memory_cycle_os.yaml` | OS memory cycle hyperparameters |
| `AgentBench/configs/memory_cycle_dbbench.yaml` | DBBench memory cycle hyperparameters |
| `MedAgentBench/configs/skill_cycle.yaml` | MedAgentBench skill cycle hyperparameters |
| `MedAgentBench/configs/memory_cycle.yaml` | MedAgentBench memory cycle hyperparameters |
| `MedAgentBench-v2/configs/skill_cycle.yaml` | MedAgentBench-v2 skill cycle hyperparameters |
| `MedAgentBench-v2/configs/memory_cycle.yaml` | MedAgentBench-v2 memory cycle hyperparameters |
| `FHIR-AgentBench/configs/skill_cycle.yaml` | FHIR-AgentBench native skill cycle hyperparameters |
| `FHIR-AgentBench/configs/memory_cycle.yaml` | FHIR-AgentBench memory cycle hyperparameters |
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

MedAgentBench-v2/skills/
└── base/               # read-only MedAgentBench-v2 base skills

FHIR-AgentBench/skills/
└── base/               # read-only FHIR-AgentBench skill template
```

Learned skills are written to `outputs/<run>/skills/learned/` during training and loaded fresh on every inference call.

Memory runs write bullets to `outputs/<run>/memory.md` and a `memory_log.jsonl` entry per update (batch index, bullets added, probe stats before and after).

---

## Implementation plan: MedAgentBench-v2 and memory cycle

### MedAgentBench-v2 (TODO)

Create `MedAgentBench-v2/` as a standalone directory mirroring MedAgentBench's structure. Copy from MedAgentBench and make the following changes:

1. **Data**: Copy `medagentbenchv2/medagentbench_v2/src/MedAgentBench/data/medagentbench/new_patient_tasks.json` to `MedAgentBench-v2/data/medagentbench/new_patient_tasks.json`. Write `split_dataset.py` with a 6/2/2 per-type split (180 dev / 60 val / 60 test).

2. **Evaluator**: Copy `medagentbenchv2/medagentbench_v2/src/medagentbenchevals/new_refsol.py` to `MedAgentBench-v2/src/server/tasks/medagentbench/refsol.py` (rename to match the existing module import path). Update `eval.py` to route all 10 task categories to the new evaluator functions.

3. **New tools**: Add `fhir_procedure_search` and `fhir_condition_search` to the agent tool set. Source the implementations from `medagentbenchv2/medagentbench_v2/src/tool/procedure_search.py` and `condition_search.py`, adapting them to the existing AgentBench-style tool interface. Register them in the agent config.

4. **Config**: Write `configs/skill_cycle.yaml` and `configs/start_task.yaml` using port 5003/5004. Point data paths at `new_patient_tasks.json`.

5. **Skill base**: Copy MedAgentBench's base skills as a starting point; update task descriptions in the skeleton to reflect the new clinical workflows.

6. **Remove `medagentbenchv2/`**: Once MedAgentBench-v2 is self-contained, `medagentbenchv2/` can be deleted.

### Memory cycle (TODO)

Implement `memory_cycle.py` (or `src/memory_cycle.py` for AgentBench-style) in each of the four benchmarks. The implementation reuses the skill cycle's epoch/batch runner, failure classification, and probe/val evaluation. Only the update step differs:

**`MemoryUpdater`** — wraps the existing `SkillUpdater` / `FHIRSkillUpdater`:
- `synthesize_bullets(failure_traces, current_bullets) -> List[str]`: calls the updater LLM with a prompt requesting 1–3 correction bullets. Prompt instructs: start each bullet with "when...", keep it ≤ 2 sentences, do not repeat existing bullets.
- Returns a plain list of strings (no validation step).

**`MemoryAwareAgent`** — wraps the base agent:
- Holds `bullets: List[str]` in memory.
- At inference time, formats bullets as a `<memory>` block and prepends it to the system prompt (AgentBench/MedAgentBench) or injects it into the FHIR-AgentBench prompt builder.
- `add_bullet(text)` appends and evicts oldest if over `max_bullets`.

**Memory cycle loop** (per benchmark):
```
for each epoch:
    for each batch:
        run dev batch with MemoryAwareAgent
        run probe with MemoryAwareAgent (log adjusted score for parity with skill logs)
        classify failures -> synthesize bullets -> add to agent
    run val with MemoryAwareAgent -> log val score
```

The probe run in the memory cycle is **logging-only** — it measures how much the latest bullet addition improved the probe, but it does not gate the update. This lets us plot "probe trajectory" comparably to skills while preserving the natural memory behavior.

**Files to create per benchmark:**

| Benchmark | Files |
|---|---|
| AgentBench | `src/memory_cycle.py`, `src/skills/memory_updater.py`, `src/client/agents/memory_aware_agent.py`, `configs/memory_cycle_{os,dbbench,ltp,card_game,alfworld}.yaml` |
| MedAgentBench | `src/memory_cycle.py`, `src/skills/memory_updater.py`, `src/client/agents/memory_aware_agent.py`, `configs/memory_cycle.yaml` |
| MedAgentBench-v2 | Same as MedAgentBench (inherited from copy) |
| FHIR-AgentBench | `memory_cycle.py`, `skill_learning/memory_updater.py`, `configs/memory_cycle.yaml` |

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

### 5. GRPO scoring — regression budget gate and symmetric error exclusion
**Files:** `AgentBench/src/skills/cycle.py`, `MedAgentBench/src/skills/cycle.py`, `FHIR-AgentBench/skill_learning/cycle.py`

Two fixes applied to all benchmarks:

**Regression budget gate:** A proposal is accepted only if `adjusted > 0 AND regressions <= baseline_regressions`. This blocks proposals that improve one failure class by trading regressions on previously passing samples.

**Symmetric error exclusion:** When a probe sample produces a task error (machinery failure, not agent failure), it is excluded from regression counting *only if that same sample also errored in the baseline probe* (`baseline_error_ids`). This prevents inflated adjusted scores from unconditionally skipping candidate errors that the baseline did not produce.

### 6. Best-checkpoint tracking
**Files:** `AgentBench/src/skills/cycle.py`, `MedAgentBench/src/skills/cycle.py`, `FHIR-AgentBench/skill_learning/cycle.py`

At the end of each epoch, if val score improves, the current `skills/learned/` directory is snapshot-copied to `skills/best/`. At run end, `skills/best/` is restored as the final skill library. This prevents the cycle from ending on a skill that helped training but hurt val.
