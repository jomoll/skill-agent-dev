# skill-agent-dev

Self-improving LLM agent framework using a GRPO-inspired skill cycle. Agents learn reusable behavioral skills from their own failures across multiple benchmarks.

```
emnlp26/
├── AgentBench/        # OS Interaction + DBBench skill cycle
└── MedAgentBench/     # FHIR medical records skill cycle
```

## How it works

After each batch of task episodes, a skill-writing LLM observes the agent's failure traces and proposes additions, modifications, or removals to a markdown skill library. Candidates are scored on a balanced probe set (fixes − regressions), and the winner is applied. Skills are injected as structured JSON context into every subsequent inference call — no fine-tuning required.

---

## Setup

### Prerequisites

- Python 3.9 (recommended)
- [Docker](https://www.docker.com/) installed and running
- Google Cloud credentials for Vertex AI (`gcloud auth application-default login`)

### 1. AgentBench (OS Interaction + DBBench)

```bash
cd AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
```

Pull and build Docker images:

```bash
# OS Interaction
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu

# DBBench — pin to MySQL 8 (MySQL 9+ removed MD5 which the benchmark requires)
docker pull mysql:8
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

---

## Running the skill cycle

### OS Interaction

```bash
# Terminal 1 — start task worker (controller on port 5001, workers from 5002)
cd AgentBench
conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_os.yaml

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_os.yaml --run-name run_001
```

### DBBench

```bash
# Terminal 1 — start task worker (controller on port 5010, workers from 5011)
cd AgentBench
conda activate agent-bench
python -m src.start_task -a --config configs/start_skill_task_dbbench.yaml --controller-port 5010 --base-port 5011

# Terminal 2 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle_dbbench.yaml --run-name run_001
```

OS and DBBench use separate controller ports (5001 and 5010) and can run in parallel.

### MedAgentBench

```bash
# Terminal 1 — start FHIR server (if not already running)
docker run -p 8080:8080 medagentbench

# Terminal 2 — start task worker
cd MedAgentBench
conda activate medagentbench
python -m src.start_task -a --config configs/start_skill_task.yaml

# Terminal 3 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle.yaml --run-name run_001
```

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
| MedAgentBench | 126 | 84 | 90 | 60/40 within tasks 1–5,8,9; tasks 6/7/10 held out |
| DBBench | 176 | 124 | 60 | 60/40 of standard.jsonl stratified by query type; dev.jsonl held out |
| OS Interaction | 79 | 56 | 35 | 60/40 of worlds 1–5,7 stratified per world; world 6 + dev.json held out |

Regenerate splits:

```bash
python AgentBench/data/dbbench/split_dataset.py
python AgentBench/data/os_interaction/split_dataset.py
python MedAgentBench/data/medagentbench/split_dataset.py
```

---

## Configuration

Key config files:

| File | Purpose |
|---|---|
| `AgentBench/configs/skill_cycle_os.yaml` | OS skill cycle hyperparameters |
| `AgentBench/configs/skill_cycle_dbbench.yaml` | DBBench skill cycle hyperparameters |
| `MedAgentBench/configs/skill_cycle.yaml` | MedAgentBench skill cycle hyperparameters |
| `AgentBench/configs/agents/gemini-chat.yaml` | Vertex AI Gemini agent config |
| `MedAgentBench/configs/agents/vertex-gemini.yaml` | Vertex AI Gemini agent config |

The agent used is Gemini Flash Lite via Vertex AI. Authentication uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) — run `gcloud auth application-default login` before starting.

---

## Skill library structure

Skills are stored as markdown files with YAML frontmatter:

```
AgentBench/skills/
├── os/base/          # read-only OS base skills
├── dbbench/base/     # read-only DBBench base skills
└── base/             # shared skeleton template

MedAgentBench/skills/
└── base/             # read-only MedAgentBench base skills
```

Learned skills are written to `outputs/<run>/skills/learned/` during training and loaded fresh on every inference call.
