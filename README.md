# HPC-AgentBench

A **trace-based benchmark** for evaluating LLM agents on real-world High-Performance Computing (HPC) support tickets.

Unlike live-environment benchmarks, HPC-AgentBench replays pre-recorded action-observation traces in a deterministic, offline simulator, so no physical cluster is required.

---

## The Dataset

HPC-AgentBench v2 uses two related schemas:

1. A **canonical schema** for dataset production, QA, and release management.
2. A **runtime projection schema** for the sandbox and evaluator.

The canonical record is the production and audit source of truth. It contains:

| Field | Description |
|---|---|
| `instance_id` | Unique identifier for the benchmark instance |
| `source_ticket_id` | Original raw ticket identifier |
| `is_valid` | Whether the sample is usable as a benchmark item |
| `release_tier` | `grounded` for main benchmark items, `reconstructed` for expansion-set items |
| `construction_mode` | `extracted` or `reconstructed` |
| `instruction` | Initial user-facing problem statement |
| `source_meta` | Raw ticket metadata such as turn count and role pattern |
| `traces` | Canonical trace objects with tool, argument, trigger, observation source, and grounding |
| `evaluation` | Reference trajectory, success criteria, and optional grounded admin reply |
| `quality` | Generator/validator metadata plus QA flags |

The runtime projection keeps the evaluator-friendly shape and includes:

| Field | Description |
|---|---|
| `instance_id` | Runtime instance identifier |
| `is_valid` | Evaluator skip flag |
| `release_tier` | Used to keep `grounded` and `reconstructed` separated |
| `construction_mode` | `extracted` or `reconstructed` |
| `instruction` | Initial observation shown to the agent |
| `traces[*].trigger_command` | Canonical tool call in `action(argument)` form |
| `traces[*].mock_output` | Simulated observation returned by the sandbox |
| `evaluation` | `expected_trajectory`, `final_solution_criteria`, `reference_admin_reply`, and `has_reference_admin_reply` |

```json
{
  "instance_id": "DEMO_001",
  "source_ticket_id": "RAW_001",
  "is_valid": true,
  "release_tier": "grounded",
  "construction_mode": "extracted",
  "instruction": "My batch job exits with code 137 and produces no useful logs. Can you help diagnose it?",
  "source_meta": {
    "turn_count": 2,
    "role_pattern": "Human>Assistant",
    "has_assistant_reply": true,
    "has_user_followup": false
  },
  "traces": [
    {
      "trace_id": "trace_001",
      "tool": "execute_bash",
      "argument": "sacct -j 4242 --format=State,ExitCode",
      "trigger_command": "execute_bash(sacct -j 4242 --format=State,ExitCode)",
      "mock_output": "State|ExitCode\nOUT_OF_MEMORY|137:0",
      "observation_source": "bash",
      "grounding": "grounded",
      "required": true
    },
    {
      "trace_id": "trace_002",
      "tool": "search_docs",
      "argument": "Slurm exit code 137 out of memory",
      "trigger_command": "search_docs(Slurm exit code 137 out of memory)",
      "mock_output": "Exit code 137 often indicates the kernel terminated the process after it exceeded its memory allocation.",
      "observation_source": "docs",
      "grounding": "grounded",
      "required": true
    }
  ],
  "evaluation": {
    "expected_trajectory": [
      "Inspect the job accounting record",
      "Confirm the meaning of exit code 137"
    ],
    "final_solution_criteria": [
      "Identify the out-of-memory failure",
      "Advise the user to request more memory or reduce usage"
    ],
    "reference_admin_reply": "Your job was terminated after exceeding its memory allocation. Please rerun with more memory.",
    "has_reference_admin_reply": true
  },
  "quality": {
    "generator_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "validator_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "qa_flags": [],
    "qa_notes": ""
  }
}
```

The example above is synthetic and documented only to illustrate the schema.

---

## The Benchmark Simulator

`HPCSandbox` (`environment.py`) is a lightweight, **order-agnostic** simulator:

1. The agent receives the ticket `instruction` as its first observation.
2. On each step the agent emits an action dict `{"action": "...", "params": {...}}`.
3. The sandbox canonicalizes tool calls into the dataset's `action(argument)` format using the benchmark's three supported tools:
   `execute_bash(command)`, `search_docs(query)`, and `ask_user_for_info(question)`.
4. Matching is done over the set of **unexecuted** traces. The sandbox first matches on tool type, then scores parameter similarity with deterministic heuristics:
   `execute_bash` emphasizes functional equivalence, while `search_docs` and `ask_user_for_info` emphasize semantic similarity.
5. If the agent selects a valid remaining tool type, the sandbox returns the best-matching trace's `mock_output` in **teacher forcing** mode, even when the parameters are only partially correct. Parameter quality is scored separately.
6. If the action is unrecognized or no remaining trace uses that tool, an error observation is returned so the agent may self-correct.
7. The episode ends when the agent calls `{"action": "reply_user", ...}`.

The interaction loop lives entirely in the **evaluator** (`evaluator.py`), not in the agent, ensuring a fair and controlled benchmark.

---

## Metrics

| Metric | Definition |
|---|---|
| **Task Success Rate** | Percentage of evaluated tickets where the agent produces a final reply judged against the ticket's `evaluation` object. By default, the evaluator uses only `grounded` items. |
| **Tool Selection Recall** | Percentage of `traces` whose tool type the agent successfully triggers per ticket, averaged across the evaluated dataset. For compatibility, the evaluator also returns this as `avg_action_recall`. |
| **Parameter Match Score** | Average quality of the matched tool arguments, measured with deterministic similarity scoring after teacher forcing picks the best remaining trace of the same tool type. |
| **Efficiency Penalty** | Average number of truly invalid actions per ticket; only actions whose tool type is absent from the remaining trace set count as errors. |

---

## Getting Started

### 1. Install dependencies

```bash
pip install openai vllm
```

### 2. Choose a local vLLM backend

Option A: run a local OpenAI-compatible vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

Option B: use in-process offline vLLM mode

No separate server is required; `run_local_benchmark.py --backend offline` will load the model directly.

### 3. Build v2 dataset artifacts from raw tickets

```bash
python process_tickets.py \
  --stage all \
  --backend offline \
  --batch-size 16 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --output pipeline_outputs
```

This writes:

- `pipeline_outputs/preclassify.jsonl`
- `pipeline_outputs/candidate.jsonl`
- `pipeline_outputs/validated.jsonl`
- `pipeline_outputs/canonical.jsonl`
- `pipeline_outputs/runtime.grounded.json`
- `pipeline_outputs/runtime.reconstructed.json`
- `pipeline_outputs/quality_report.json`

The outputs above are the full candidate artifacts produced by the pipeline.
They are suitable for analysis, auditing, and local benchmark runs, but they
are not yet the final release package for a public benchmark leaderboard.

### 4. Sample 100 tickets and run the local benchmark

```bash
python run_local_benchmark.py \
  --backend offline \
  --sample-size 100 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

Or, if you already have a local vLLM server running:

```bash
python run_local_benchmark.py \
  --backend server \
  --sample-size 100 \
  --server-base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

The runner shares the same local model backend between the baseline agent and the LLM judge. The evaluator skips any records where `is_valid` is `false` and, by default, only scores `release_tier=grounded`.

### 5. Release Curation

The current pipeline is designed to generate a **candidate full dataset**. In
practice, we recommend one additional deterministic curation step before
publishing a release:

- Start from `canonical.jsonl` as the full candidate corpus
- Keep `runtime.reconstructed.json` as the expansion split
- Filter `grounded` more aggressively using the existing QA flags so that the
  public main benchmark split is high precision rather than maximally large

In other words:

- `canonical.jsonl` is the audit and production source of truth
- `runtime.*.json` files are benchmark-ready projections
- the final public `grounded` release should be a curated subset of the raw
  pipeline output, not a blind export of every candidate labeled `grounded`

### 6. Quick smoke test (no API key needed)

```bash
python evaluator.py
```

This runs the built-in `_SmokeTestAgent` on a synthetic ticket in the new schema and prints per-ticket and aggregate metrics.

### 7. Audit an existing cleaned dataset

```bash
python process_tickets.py \
  --audit-cleaned backup/benchmark_results.cleaned.json \
  --audit-output cleaned_audit_report.json
```

The audit scans for traces that introduce hidden usernames, job IDs, absolute paths, or resource values before those entities have been surfaced through the instruction, an earlier `ask_user_for_info` reply, or a prior `mock_output`.

---

## Project Structure

```text
HPC-AgentBench/
├── interface.py                           # BaseHPCAgent abstract class
├── environment.py                         # HPCSandbox runtime simulator
├── evaluator.py                           # Evaluation loop and metrics
├── baseline_agent.py                      # Local vLLM baseline agent
├── local_llm.py                           # Shared local inference client
├── dataset_schema.py                      # Canonical schema, validation, projection, reports
├── benchmark_semantics.py                 # Semantic matching and audit helpers
├── process_tickets.py                     # Multi-stage raw -> canonical/runtime pipeline
├── run_local_benchmark.py                 # Sample runtime dataset and evaluate locally
├── backup/benchmark_results.cleaned.json  # Legacy cleaned dataset snapshot
└── README.md
```
