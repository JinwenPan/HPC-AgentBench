"""
Evaluator – the main benchmark engine.

Houses the interaction loop and metric calculations.  The agent never
drives its own loop; the evaluator calls ``agent.take_action()`` and
feeds the result through ``env.step()`` in a controlled while-loop.
"""

from typing import List, Dict, Any

from interface import BaseHPCAgent
from environment import HPCSandbox


# ------------------------------------------------------------------ #
#  Dummy LLM-as-a-Judge (placeholder)                                #
# ------------------------------------------------------------------ #

def llm_judge(agent_reply: str, ground_truth_resolution: str) -> bool:
    """
    Evaluate whether the agent's final reply resolves the ticket.

    Current implementation: returns ``True`` whenever the agent called
    ``reply_user`` (i.e. *agent_reply* is non-empty).  Replace with an
    actual LLM grading call later.
    """
    return bool(agent_reply)


# ------------------------------------------------------------------ #
#  Core evaluation function                                           #
# ------------------------------------------------------------------ #

def evaluate_agent(
    agent: BaseHPCAgent,
    dataset: List[Dict[str, Any]],
    max_steps: int = 15,
) -> Dict[str, Any]:
    """
    Run *agent* on every ticket in *dataset* and compute aggregate metrics.

    Args:
        agent:     An instance that implements :class:`BaseHPCAgent`.
        dataset:   List of ticket dicts, each containing at least
                   ``initial_user_query`` and ``required_actions``.
        max_steps: Maximum interaction steps per ticket before giving up.

    Returns:
        A dict with keys ``task_success_rate``, ``avg_action_recall``,
        and ``avg_efficiency_penalty``.
    """
    total_tickets = len(dataset)
    if total_tickets == 0:
        print("Dataset is empty. Exiting.")
        return {}

    successes = 0
    total_action_recall_pct = 0.0
    total_errors = 0

    print(f"Starting evaluation on {total_tickets} ticket(s)...\n")

    for i, ticket in enumerate(dataset):
        ticket_id = ticket.get("ticket_id", f"Unknown_{i}")
        print(f"--- Ticket {i + 1}/{total_tickets}: {ticket_id} ---")

        # Reset agent state for a fresh interaction
        agent.reset()

        # Initialise sandbox
        sandbox = HPCSandbox(ticket)

        # Build the initial observation the agent sees
        initial_query = ticket.get("initial_user_query", "")
        metadata = ticket.get("metadata", {})
        observation = f"User Query: {initial_query}\nMetadata: {metadata}"

        done = False
        step_count = 0
        errors_this_ticket = 0
        agent_reply = ""

        # ---- Interaction loop (lives HERE, not in the agent) ----
        while not done and step_count < max_steps:
            action = agent.take_action(observation)
            observation, done, info = sandbox.step(action)

            if not done and info.get("status") == "error":
                errors_this_ticket += 1

            if done:
                agent_reply = observation

            step_count += 1

        if not done:
            print(
                f"  [Warning] Max steps ({max_steps}) reached without "
                "'reply_user'."
            )

        # ---- Per-ticket metrics ----
        # 1. Task Success
        gt_resolution = ticket.get("resolution_summary", "")
        success = llm_judge(agent_reply, gt_resolution)
        if success:
            successes += 1

        # 2. Action Recall
        required_count = len(sandbox.required_actions)
        if required_count > 0:
            executed_count = sum(sandbox.executed_actions)
            recall_pct = (executed_count / required_count) * 100.0
        else:
            recall_pct = 100.0

        total_action_recall_pct += recall_pct

        # 3. Efficiency Penalty
        total_errors += errors_this_ticket

        print(f"  Steps:          {step_count}")
        print(f"  Invalid/Errors: {errors_this_ticket}")
        print(f"  Action Recall:  {recall_pct:.2f}%")
        print(f"  Task Success:   {success}\n")

    # ---- Aggregate metrics ----
    avg_success_rate = (successes / total_tickets) * 100.0
    avg_action_recall = total_action_recall_pct / total_tickets
    avg_efficiency_penalty = total_errors / total_tickets

    print("=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total Tickets Evaluated:  {total_tickets}")
    print(f"Task Success Rate:        {avg_success_rate:.2f}%")
    print(f"Average Action Recall:    {avg_action_recall:.2f}%")
    print(f"Efficiency Penalty:       {avg_efficiency_penalty:.2f} errors/ticket")

    return {
        "task_success_rate": avg_success_rate,
        "avg_action_recall": avg_action_recall,
        "avg_efficiency_penalty": avg_efficiency_penalty,
    }


# ------------------------------------------------------------------ #
#  Inline test agent + smoke test                                     #
# ------------------------------------------------------------------ #

class _RandomAgent(BaseHPCAgent):
    """Deterministic dummy agent used only for integration testing."""

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def take_action(self, observation: str) -> Dict[str, Any]:
        self._step += 1
        if self._step == 1:
            return {"action": "get_job_state", "params": {"user_id": "alice"}}
        if self._step == 2:
            return {"action": "check_syslog", "params": {"node": "gpu01"}}
        return {
            "action": "reply_user",
            "params": {"text": "The job was OOM killed."},
        }


if __name__ == "__main__":
    dummy_dataset = [
        {
            "ticket_id": "HPC_001",
            "initial_user_query": "My python job crashed without any output.",
            "metadata": {"user_id": "alice", "node": "gpu01"},
            "required_actions": [
                {
                    "expected_action": "get_job_state",
                    "expected_params": {"user_id": "alice"},
                    "mock_observation": "Job 1024 FAILED. ExitCode 137.",
                },
                {
                    "expected_action": "check_syslog",
                    "expected_params": {"node": "gpu01"},
                    "mock_observation": (
                        "Out of memory: Killed process python."
                    ),
                },
            ],
            "resolution_summary": (
                "The job was killed due to OOM. "
                "User needs to request more memory."
            ),
        }
    ]

    print("Running integration test with _RandomAgent...\n")
    evaluate_agent(_RandomAgent(), dummy_dataset)
