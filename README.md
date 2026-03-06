# HPC-AgentBench

A **trace-based benchmark** for evaluating LLM agents on real-world High-Performance Computing (HPC) support tickets.

Unlike live-environment benchmarks, HPC-AgentBench replays pre-recorded action–observation traces in a deterministic, offline simulator — no physical cluster required.

---

## The Dataset

Each ticket is a JSON object distilled from a real HPC helpdesk interaction.
A ticket captures:

| Field | Description |
|---|---|
| `ticket_id` | Unique identifier |
| `initial_user_query` | The verbatim user message that opens the ticket |
| `metadata` | Contextual information (user ID, node, job ID, etc.) |
| `required_actions` | **Trace graph** — an unordered set of actions the expert performed, each with `expected_action`, `expected_params`, and a `mock_observation` |
| `resolution_summary` | Ground-truth resolution text |

```json
{
  "ticket_id": "HPC_001",
  "initial_user_query": "My python job crashed without any output.",
  "metadata": {"user_id": "alice", "node": "gpu01"},
  "required_actions": [
    {
      "expected_action": "get_job_state",
      "expected_params": {"user_id": "alice"},
      "mock_observation": "Job 1024 FAILED. ExitCode 137."
    },
    {
      "expected_action": "check_syslog",
      "expected_params": {"node": "gpu01"},
      "mock_observation": "Out of memory: Killed process python."
    }
  ],
  "resolution_summary": "The job was killed due to OOM. User needs to request more memory."
}
```

---

## The Benchmark Simulator

`HPCSandbox` (`environment.py`) is a lightweight, **order-agnostic** simulator:

1. The agent receives the `initial_user_query` as its first observation.
2. On each step the agent emits an action dict `{"action": "...", "params": {...}}`.
3. If the action matches any **unexecuted** entry in `required_actions`, the sandbox returns its `mock_observation`.
4. If the action is unrecognised, an error observation is returned — the agent may self-correct.
5. The episode ends when the agent calls `{"action": "reply_user", ...}`.

The interaction loop lives entirely in the **evaluator** (`evaluator.py`), not in the agent, ensuring a fair and controlled benchmark.

---

## Metrics

| Metric | Definition |
|---|---|
| **Task Success Rate** | Percentage of tickets where the agent's final reply resolves the issue (judged by an LLM-as-a-Judge). |
| **Action Recall** | Percentage of `required_actions` the agent successfully triggered per ticket, averaged across the dataset. |
| **Efficiency Penalty** | Average number of invalid/error actions per ticket — lower is better. |

---

## Getting Started

### 1. Install dependencies

```bash
pip install openai
```

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the baseline agent

```python
import json
from baseline_agent import NaiveOpenAIAgent
from evaluator import evaluate_agent

with open("my_dataset.json") as f:
    dataset = json.load(f)

agent = NaiveOpenAIAgent(model="gpt-4o-mini")
results = evaluate_agent(agent, dataset)
```

### 4. Quick smoke test (no API key needed)

```bash
python evaluator.py
```

This runs the built-in `_RandomAgent` on a dummy ticket and prints per-ticket and aggregate metrics.

---

## Project Structure

```
HPC-AgentBench/
├── interface.py        # BaseHPCAgent abstract class (the contract)
├── environment.py      # HPCSandbox trace-based simulator
├── evaluator.py        # Main evaluation loop & metrics
├── baseline_agent.py   # NaiveOpenAIAgent baseline
└── README.md
```
