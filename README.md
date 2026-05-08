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

### Memory comparators

Each benchmark ships memory-based comparators alongside the `skill_cycle`, answering three questions: (1) does structured skill encoding beat cheap natural-language memory, (2) does the batch vs. per-sample update cadence matter, and (3) does retrieved Evo-style episodic/semantic memory close the gap to validated skill writing?

**`memory_cycle` (sequential, paper-faithful)** — matches the MedAgentBench-v2 paper (Appendix A.2). After each individual failing sample, the updater LLM is called once with the paper's prompt template:

```
<task_description>  {instruction + context}
<agent_response>    {agent's final answer}
<eval_output>       {ref_sol + pass/fail}
<current_prompt>    {current <memory> block}
```

Output is plain prose starting with `"when asked ..."` — task-specific conditional instructions, not general rules. Dev samples are run sequentially (no parallelism); val evaluation is still parallelised.

**`batch_memory_cycle`** — parallel batches of `update_every` samples; memory updated once per batch from all failing traces in that batch. Matches our skill cycle's batch cadence for a direct apples-to-apples comparison. Memory notes use the same paper-style prompt but process multiple failing entries per update call (one LLM call per entry).

**`evo_memory_cycle`** — Evo-Memory-style structured memory comparator. Instead of appending all notes into one prompt, it maintains two stores: episodic memories of completed task attempts and a compact semantic cheatsheet of reusable procedural rules. At inference time it retrieves only top-k relevant rules/episodes; after each dev episode, a curator LLM reflects with eval feedback, updates the stores, and tracks rule utility (`shown`, `success`, `failure`). There is no probe acceptance gate — this tests retrieval and utility-tracked memory curation against validated skill writing.

Note: the Evo comparator's baseline is a **protocol-only baseline**. Even before any episodic or semantic memory exists, the agent receives the Evo memory-guided reasoning protocol ("use memory as strategy guidance, not answer lookup; prefer current task details; reuse portable workflows"). Later epochs add retrieved rules/episodes on top of that fixed protocol, so Evo learning curves should be interpreted as memory accumulation relative to the Evo protocol baseline, not as a prompt-identical baseline to `memory_cycle`.

**`skillx_cycle`** — SkillX extraction-based comparator (arXiv 2604.04804, ZJUNLP/Ant Group). Instead of editing skills in response to failures, it distills *successful* trajectories into a hierarchical skill library using a two-stage pipeline: (1) a `FunctionalSkillExtractor` LLM call decomposes each successful trace into step-level reusable skills; (2) a `TwoStageFilterPipeline` keeps only high-signal skills (general quality filter; tool-schema validation skipped as it requires benchmark-specific schemas). Extracted skills are merged into a persistent `skillx_library.json` after each epoch. At inference, top-k skills are retrieved via BM25 overlap and injected as a `<skillx_memory>` block. Available in MedAgentBench, MedAgentBench-v2, and FHIR-AgentBench. Requires the SkillX repo at `/home/moll/skill-agent-dev/SkillX` (configured via `skillx.skillx_dir`).

| Comparator | Learned artifact | Update source | Selection mechanism |
|---|---|---|---|
| `skill_cycle` | Markdown skills | Failure traces | Probe-scored fixes minus regressions |
| `memory_cycle` | Flat correction notes | Individual failures | Append all |
| `batch_memory_cycle` | Flat correction notes | Batch failures | Append all |
| `evo_memory_cycle` | Episodic examples + semantic rules | Completed dev episodes with eval feedback | Top-k retrieval + rule utility |
| `skillx_cycle` | Hierarchical functional skill library | Successful dev episodes | BM25 retrieval |

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
| MedAgentBench | 5050 | 5051 |
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
python -m src.start_task -a --config configs/start_task.yaml --base-port 5071

# Terminal 3 — run skill cycle
python -m src.skill_cycle --config configs/skill_cycle.yaml --run-name run_001
```

### FHIR-AgentBench

FHIR-AgentBench does not use the AgentBench task controller. Its skill cycle wraps
the existing FHIR agents directly, injects learned markdown skills into each agent's
system prompt, and scores samples with a cached per-sample answer judge.

```bash
cd FHIR-AgentBench && conda activate fhir-agentbench

# Text/resource agent
python skill_cycle.py --config configs/skill_cycle_text.yaml --run-name run_001

# Code/resource agent
python skill_cycle.py --config configs/skill_cycle_code.yaml --run-name run_001
```

If a run is interrupted or killed, resume it from the last completed samples:

```bash
python skill_cycle.py --config configs/skill_cycle_text.yaml --run-name run_001 --resume
python skill_cycle.py --config configs/skill_cycle_code.yaml --run-name run_001 --resume
```

`configs/skill_cycle_text.yaml` uses `multi_turn_resource`; `configs/skill_cycle_code.yaml`
uses `multi_turn_code_resource`. Both use `openai/gpt-oss-120b`, the CSV at
`final_dataset/questions_answers_sql_fhir.csv`, the CSV `train` split as dev, and
the CSV `valid` split as val. For local vLLM/LiteLLM-compatible models, set
`agent.base_url`, `updater.base_url`, and `eval.base_url` in the selected config.

FHIR-AgentBench uses the same grouped proposal-ranking shape as AgentBench and
MedAgentBench: `cycle.grpo_k` is the total number of proposal calls per update,
cycled over the largest failure modes, and each validated proposal is ranked on
the same probe set against current-skill baseline fixes/regressions before a
single winner is applied.

### Evaluating with a manual skill pack (AgentBench)

To run a fixed set of skills against a split (useful as an upper-bound reference):

```bash
cd AgentBench
python -m src.run_manual_skills --config configs/manual_skills_dbbench.yaml --split val
```

### Evaluating on the held-out test set (MedAgentBench, MedAgentBench-v2, and FHIR-AgentBench)

`src/run_eval.py` evaluates either the base agent (no learned skills) or a saved skill pack against any split. Use this after a skill cycle run to get test-set numbers.

```bash
# MedAgentBench
cd MedAgentBench && conda activate medagentbench

# Base agent on test set
python -m src.run_eval --config configs/skill_cycle.yaml --split test --run-name base_test

# Best skills from a completed run on test set
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir outputs/skill_cycle/run_001/skills/best --run-name run_001_best_test
```

```bash
# MedAgentBench-v2
cd MedAgentBench-v2 && conda activate medagentbench

# Base agent on test set
python -m src.run_eval --config configs/skill_cycle.yaml --split test --run-name base_test

# Best skills from a completed run on test set
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir outputs/skill_cycle/run_001/skills/best --run-name run_001_best_test
```

The task worker must be running before invoking `run_eval.py` (same startup as the skill cycle). Results are written to `outputs/eval/<run-name>/`:

- `test_runs.jsonl` — per-sample correctness and task result
- `test_score.json` — summary `{split, score, n_correct, n_total, skills_dir}`

`--split` accepts `dev`, `val`, or `test`. Omit `--skills-dir` for the base agent; pass a path to any `skills/learned/` or `skills/best/` directory for a skills-equipped agent. Use `--force` to overwrite an existing run directory.

```bash
# FHIR-AgentBench (no task worker needed — agents run in-process)
cd FHIR-AgentBench && conda activate fhir-agentbench

# Base agent on test set (code agent)
python run_eval.py --config configs/skill_cycle_code.yaml --split test --run-name base_test

# Best skills from a completed run on test set (code agent)
python run_eval.py --config configs/skill_cycle_code.yaml --split test \
    --skills-dir outputs/skill_cycle_code/run_001/skills/best --run-name run_001_best_test

# Same pattern for the text agent
python run_eval.py --config configs/skill_cycle_text.yaml --split test --run-name base_test_text
```

Results are written to `outputs/eval/<run-name>/`. FHIR-AgentBench uses an LLM judge for non-exact-match answers; scores are cached in `outputs/eval/<run-name>/eval_cache.json`.

> **Note:** FHIR-AgentBench eval runs occasionally get stuck (e.g. due to a hung agent or network timeout). If this happens, kill the process and resume from where it left off — completed samples are written incrementally to the JSONL so nothing is lost:
> ```bash
> python run_eval.py --config configs/skill_cycle_code.yaml --split test \
>     --run-name run_001_best_test --resume
> ```

### Cross-benchmark skill transfer (FHIR triplet)

Skills learned on one FHIR benchmark can be evaluated on another without any code changes — the skill file format (Markdown + YAML frontmatter) is identical across MedAgentBench, MedAgentBench-v2, and FHIR-AgentBench. Pass `--skills-dir` pointing to the source benchmark's `skills/best/` directory and run `run_eval.py` for the target benchmark as normal. The target benchmark's base skills are still loaded alongside the transferred ones.

The task worker for the target benchmark must be running (same as a normal eval). FHIR-AgentBench is self-contained and needs no worker.

**MedAgentBench → MedAgentBench-v2**

```bash
cd MedAgentBench-v2 && conda activate medagentbench
python -m src.run_eval --config configs/skill_cycle.yaml --split val \
    --skills-dir ../MedAgentBench/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_val
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir ../MedAgentBench/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_test
```

**MedAgentBench-v2 → MedAgentBench**

```bash
cd MedAgentBench && conda activate medagentbench
python -m src.run_eval --config configs/skill_cycle.yaml --split val \
    --skills-dir ../MedAgentBench-v2/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_v2_val
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir ../MedAgentBench-v2/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_v2_test
```

**MedAgentBench → FHIR-AgentBench**

```bash
cd FHIR-AgentBench && conda activate fhir-agentbench
python run_eval.py --config configs/skill_cycle_code.yaml --split val \
    --skills-dir ../MedAgentBench/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_val
python run_eval.py --config configs/skill_cycle_code.yaml --split test \
    --skills-dir ../MedAgentBench/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_test
# repeat with --config configs/skill_cycle_text.yaml for the text agent
```

**MedAgentBench-v2 → FHIR-AgentBench**

```bash
cd FHIR-AgentBench && conda activate fhir-agentbench
python run_eval.py --config configs/skill_cycle_code.yaml --split val \
    --skills-dir ../MedAgentBench-v2/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_v2_val
python run_eval.py --config configs/skill_cycle_code.yaml --split test \
    --skills-dir ../MedAgentBench-v2/outputs/skill_cycle/run_001/skills/best \
    --run-name cross_from_mab_v2_test
# repeat with --config configs/skill_cycle_text.yaml for the text agent
```

**FHIR-AgentBench → MedAgentBench**

```bash
cd MedAgentBench && conda activate medagentbench
# skills from the code agent
python -m src.run_eval --config configs/skill_cycle.yaml --split val \
    --skills-dir ../FHIR-AgentBench/outputs/skill_cycle_code/run_001/skills/best \
    --run-name cross_from_fab_code_val
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir ../FHIR-AgentBench/outputs/skill_cycle_code/run_001/skills/best \
    --run-name cross_from_fab_code_test
# repeat with skill_cycle_text/run_001/skills/best for the text agent
```

**FHIR-AgentBench → MedAgentBench-v2**

```bash
cd MedAgentBench-v2 && conda activate medagentbench
# skills from the code agent
python -m src.run_eval --config configs/skill_cycle.yaml --split val \
    --skills-dir ../FHIR-AgentBench/outputs/skill_cycle_code/run_001/skills/best \
    --run-name cross_from_fab_code_val
python -m src.run_eval --config configs/skill_cycle.yaml --split test \
    --skills-dir ../FHIR-AgentBench/outputs/skill_cycle_code/run_001/skills/best \
    --run-name cross_from_fab_code_test
# repeat with skill_cycle_text/run_001/skills/best for the text agent
```

---

## Running the memory cycle

Both memory comparators share all infrastructure with the skill cycle (task workers, ports, FHIR server). No additional setup is needed.

### Sequential memory cycle (paper-faithful)

```bash
# AgentBench (OS Interaction)
python -m src.memory_cycle --config configs/memory_cycle_os.yaml --run-name mem_001

# AgentBench (DBBench)
python -m src.memory_cycle --config configs/memory_cycle_dbbench.yaml --run-name mem_001

# MedAgentBench
python -m src.memory_cycle --config configs/memory_cycle.yaml --run-name mem_001

# MedAgentBench-v2
python -m src.memory_cycle --config configs/memory_cycle.yaml --run-name mem_001

# FHIR-AgentBench
python memory_cycle.py --config configs/memory_cycle.yaml --run-name mem_001
```

### Batch memory cycle

```bash
# AgentBench (OS Interaction)
python -m src.batch_memory_cycle --config configs/batch_memory_cycle_os.yaml --run-name mem_001

# AgentBench (DBBench)
python -m src.batch_memory_cycle --config configs/batch_memory_cycle_dbbench.yaml --run-name mem_001

# MedAgentBench
python -m src.batch_memory_cycle --config configs/batch_memory_cycle.yaml --run-name mem_001

# MedAgentBench-v2
python -m src.batch_memory_cycle --config configs/batch_memory_cycle.yaml --run-name mem_001

# FHIR-AgentBench
python batch_memory_cycle.py --config configs/batch_memory_cycle.yaml --run-name mem_001
```

### Evo memory cycle

```bash
# AgentBench
python -m src.evo_memory_cycle --config configs/evo_memory_cycle_os.yaml --run-name evo_001
python -m src.evo_memory_cycle --config configs/evo_memory_cycle_dbbench.yaml --run-name evo_001
python -m src.evo_memory_cycle --config configs/evo_memory_cycle_ltp.yaml --run-name evo_001
python -m src.evo_memory_cycle --config configs/evo_memory_cycle_card_game.yaml --run-name evo_001
python -m src.evo_memory_cycle --config configs/evo_memory_cycle_alfworld.yaml --run-name evo_001

# MedAgentBench
python -m src.evo_memory_cycle --config configs/evo_memory_cycle.yaml --run-name evo_001

# MedAgentBench-v2
python -m src.evo_memory_cycle --config configs/evo_memory_cycle.yaml --run-name evo_001

# FHIR-AgentBench
python evo_memory_cycle.py --config configs/evo_memory_cycle.yaml --run-name evo_001
```

**Design principles for a fair comparison:**
- Same base agent and updater model as the skill cycle — no enhanced system prompt, no additional tools
- Same dev/val splits and val evaluation
- Memory notes appended unconditionally (no probe-based acceptance gate) — the gate is one of the variables under test
- Val score tracked identically, enabling direct learning-curve comparison
- Evo memory also avoids the probe gate; its distinct variable is retrieved structured memory with utility-tracked semantic rules
- Evo memory uses a protocol-only baseline, so compare epoch gains against its own baseline; compare absolute baseline values to other approaches with that prompt difference in mind

**Implementation across benchmarks:**

| Benchmark | Memory agent | Memory storage | Note synthesis |
|---|---|---|---|
| AgentBench / MedAgentBench / MedAgentBench-v2 | `MemoryAwareAgent` wrapping base agent | `memory.json` (flat JSON list) injected as `<memory>` block | One LLM call per failing sample, paper prompt format |
| FHIR-AgentBench | Memory block prepended to agent `system_msg` per run | `memory.json` in run dir | Same paper prompt format |

| Benchmark | Evo memory agent | Evo storage | Retrieval/update |
|---|---|---|---|
| AgentBench / MedAgentBench / MedAgentBench-v2 | `EvoMemoryAwareAgent` wrapping base agent | `evo_memory/episodic.jsonl`, `evo_memory/semantic.json` | Lexical top-k retrieval + curator reflection after each dev episode |
| FHIR-AgentBench | Native system-prompt wrapper | same | Native runner integration with the same curator schema |

The `update_every` and `batch_concurrency` keys are used only by the batch variant; the sequential variant ignores `update_every` (updates after every failure) but still uses `batch_concurrency` for parallelised val evaluation. Memory grows unbounded — no condensing, matching the original paper.

---

### SkillX cycle

Requires the SkillX repo cloned alongside the benchmarks (`/home/moll/skill-agent-dev/SkillX`). Set `skillx.skillx_dir` in the config to the repo root. Not available for AgentBench (no sequential run structure for epoch-level extraction).

```bash
# MedAgentBench
python -m src.skillx_cycle --config configs/skillx_cycle.yaml --run-name skillx_001

# MedAgentBench-v2
python -m src.skillx_cycle --config configs/skillx_cycle.yaml --run-name skillx_001

# FHIR-AgentBench
python skillx_cycle.py --config configs/skillx_cycle.yaml --run-name skillx_001
```

Output per epoch includes `epoch_N/skillx_updates.json` (`n_successful`, `n_extracted`, `n_filtered`, `n_after_merge`) and a shared `skillx_library.json` in the run root. Val scores are tracked in `val_scores.json` matching all other comparators.

| Benchmark | Skill agent | Skill storage | Extraction trigger |
|---|---|---|---|
| MedAgentBench / MedAgentBench-v2 | `SkillXAwareAgent` wrapping base agent | `skillx_library.json` injected as `<skillx_memory>` block | Once per epoch on all successful dev traces |
| FHIR-AgentBench | Skill block prepended to agent `system_msg` per run | same | Once per epoch on all successful dev traces |

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
| `AgentBench/configs/memory_cycle_os.yaml` | OS sequential memory cycle |
| `AgentBench/configs/memory_cycle_dbbench.yaml` | DBBench sequential memory cycle |
| `AgentBench/configs/batch_memory_cycle_os.yaml` | OS batch memory cycle |
| `AgentBench/configs/batch_memory_cycle_dbbench.yaml` | DBBench batch memory cycle |
| `AgentBench/configs/evo_memory_cycle_*.yaml` | Evo-style structured memory comparators |
| `MedAgentBench/configs/skill_cycle.yaml` | MedAgentBench skill cycle hyperparameters |
| `MedAgentBench/configs/memory_cycle.yaml` | MedAgentBench sequential memory cycle |
| `MedAgentBench/configs/batch_memory_cycle.yaml` | MedAgentBench batch memory cycle |
| `MedAgentBench/configs/evo_memory_cycle.yaml` | MedAgentBench Evo memory comparator |
| `MedAgentBench-v2/configs/skill_cycle.yaml` | MedAgentBench-v2 skill cycle hyperparameters |
| `MedAgentBench-v2/configs/memory_cycle.yaml` | MedAgentBench-v2 sequential memory cycle |
| `MedAgentBench-v2/configs/batch_memory_cycle.yaml` | MedAgentBench-v2 batch memory cycle |
| `MedAgentBench-v2/configs/evo_memory_cycle.yaml` | MedAgentBench-v2 Evo memory comparator |
| `FHIR-AgentBench/configs/skill_cycle_text.yaml` | FHIR-AgentBench text/resource-agent skill cycle |
| `FHIR-AgentBench/configs/skill_cycle_code.yaml` | FHIR-AgentBench code/resource-agent skill cycle |
| `FHIR-AgentBench/configs/memory_cycle.yaml` | FHIR-AgentBench sequential memory cycle |
| `FHIR-AgentBench/configs/batch_memory_cycle.yaml` | FHIR-AgentBench batch memory cycle |
| `FHIR-AgentBench/configs/evo_memory_cycle.yaml` | FHIR-AgentBench Evo memory comparator |
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

Memory runs write bullets to `outputs/<run>/memory.json` and per-epoch update logs as `memory_updates.json`.

Evo memory runs write structured stores to `outputs/<run>/evo_memory/episodic.jsonl` and `outputs/<run>/evo_memory/semantic.json`; per-episode curator updates are logged under each epoch as `evo_memory_updates.json`.

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

### 6. Lenient result scoring — format-independent numeric matching (MedAgentBench)
**Files:** `MedAgentBench/src/server/tasks/medagentbench/utils.py`, `MedAgentBench/src/server/tasks/medagentbench/refsol.py`

The original graders called `json.loads(results.result)` which silently returns `False` via a bare `except` when `results.result` is already a Python object (list or number) rather than a JSON string — a frequent occurrence when results travel through the HTTP task API. This caused false negatives on all read tasks.

Additionally, read-task graders required exact JSON equality for numeric answers. Agents that found the correct value but expressed it as prose (e.g., `"85 mg/dL"` instead of `[85.0]`) were scored incorrect even though the clinical answer was right, making base-agent vs. skill-agent comparisons confounded by formatting rather than reasoning.

Two helpers added to `utils.py`:
- `parse_agent_result(raw)` — normalises to Python object whether `raw` is a JSON string or already deserialized.
- `extract_numeric(raw)` — extracts the first numeric value from any result format (Python list, list of prose strings, bare string).
- `match_agent_result(ref_sol, raw, tol=0.0, accept_empty=False)` — tries exact match first, then numeric extraction with configurable tolerance (default 0, i.e. exact numeric value regardless of units or surrounding text). `accept_empty=True` for write tasks that accept `[]` as a valid completion signal.

All read-task graders (`task1`, `task2`, `task4`, `task5`, `task6`, `task7`, `task9`, `task10`) updated to use these helpers. Write-task graders (`task3`, `task8`) are unchanged (they check FHIR POST history, not `results.result`).

### 8. Best-checkpoint tracking
**Files:** `AgentBench/src/skills/cycle.py`, `MedAgentBench/src/skills/cycle.py`, `FHIR-AgentBench/skill_learning/cycle.py`

At the end of each epoch, if val score improves, the current `skills/learned/` directory is snapshot-copied to `skills/best/`. At run end, `skills/best/` is restored as the final skill library. This prevents the cycle from ending on a skill that helped training but hurt val.

### 9. Evo memory comparator — retrieved episodic + semantic memory
**Files:**
- `AgentBench/src/evo_memory/`, `AgentBench/src/evo_memory_cycle.py`
- `AgentBench/src/client/agents/evo_memory_aware_agent.py`
- `MedAgentBench/src/evo_memory/`, `MedAgentBench/src/evo_memory_cycle.py`
- `MedAgentBench/src/client/agents/evo_memory_aware_agent.py`
- `MedAgentBench-v2/src/evo_memory/`, `MedAgentBench-v2/src/evo_memory_cycle.py`
- `MedAgentBench-v2/src/client/agents/evo_memory_aware_agent.py`
- `FHIR-AgentBench/evo_memory/`, `FHIR-AgentBench/evo_memory_cycle.py`
- `FHIR-AgentBench/skill_learning/evo_memory_cycle.py`

An Evo-Memory-style comparator is added alongside the flat memory comparators. It stores completed dev episodes in `evo_memory/episodic.jsonl` and compact reusable procedural rules in `evo_memory/semantic.json`.

- **Retrieved context:** inference injects only top-k relevant semantic rules and episodic summaries, rather than the full memory store.
- **Semantic utility tracking:** each rule tracks `shown`, `success`, and `failure`; retrieval ranks by priority, lexical relevance, and a UCB-style utility score.
- **Curator updates:** after every non-error dev episode, the updater LLM reflects with eval feedback and emits `episodic_summary`, `failure_analysis`, `action_guidelines`, and `tags`.
- **No probe gate:** unlike `skill_cycle`, Evo memory updates are not accepted or rejected via probe scoring. This keeps the comparator focused on retrieved structured memory rather than validated skill mutation.
- **Injection point:** AgentBench-style repos use `EvoMemoryAwareAgent`, mirroring skill/memory injection: prefix on the first decision and suffix on continuation turns. FHIR-AgentBench injects the retrieved memory block into the native agent system prompt.

### 10. SkillX comparator — extraction-based functional skill library
**Files:**
- `MedAgentBench/src/skillx/`, `MedAgentBench/src/skillx_cycle.py`, `MedAgentBench/configs/skillx_cycle.yaml`
- `MedAgentBench-v2/src/skillx/`, `MedAgentBench-v2/src/skillx_cycle.py`, `MedAgentBench-v2/configs/skillx_cycle.yaml`
- `FHIR-AgentBench/skill_learning/skillx/`, `FHIR-AgentBench/skill_learning/skillx_cycle.py`
- `FHIR-AgentBench/skillx_cycle.py`, `FHIR-AgentBench/configs/skillx_cycle.yaml`

SkillX (arXiv 2604.04804) distills successful agent trajectories into a hierarchical skill library via LLM extraction rather than GRPO-style editing. After every epoch, all successful dev traces are passed to a `FunctionalSkillExtractor` which decomposes each trace into step-level pseudocode skills (one LLM call per plan step). A `TwoStageFilterPipeline` removes low-quality skills (general quality filter; tool-schema stage 2 skipped). Filtered skills are merged into a `SkillLibrary` via `library.merge()` (update existing by name, add new). At inference, top-k skills are retrieved by BM25 overlap and injected as a `<skillx_memory>` block.

The SkillX `SkillX/pipeline.py` entry point is never imported (it requires `langchain_openai`). Only the submodules (`extraction/`, `filtering/`, `core/`) are used directly, with `SkillXLLMAdapter` bridging the sync agent to the async `ainvoke()` interface they expect.
