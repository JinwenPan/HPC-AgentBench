"""
Evaluator – the main benchmark engine.

Houses the interaction loop and metric calculations.  The agent never
drives its own loop; the evaluator calls ``agent.take_action()`` and
feeds the result through ``env.step()`` in a controlled while-loop.
"""

import json
from pprint import pformat
from typing import Any, Dict, List
from typing import Optional

from interface import BaseHPCAgent
from environment import HPCSandbox
from local_llm import LocalLLMClient

JUDGE_SYSTEM_PROMPT = """\
You are a strict benchmark judge for HPC support responses.

Decide whether the agent's final reply satisfies the required solution
criteria for the ticket. Consider the reference admin reply as supporting
context, but judge primarily on whether the response meets the stated
criteria and resolves the user's issue.

Return only valid JSON in this format:
{"passed": true, "reason": "short explanation"}
"""


# ------------------------------------------------------------------ #
#  Dummy LLM-as-a-Judge (placeholder)                                #
# ------------------------------------------------------------------ #

def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _parse_judge_verdict(raw_text: str) -> Optional[bool]:
    cleaned = _strip_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        normalized = cleaned.lower().strip()
        if normalized in {"true", "yes", "pass", "passed"}:
            return True
        if normalized in {"false", "no", "fail", "failed"}:
            return False
        if '"passed": true' in normalized:
            return True
        if '"passed": false' in normalized:
            return False
        return None

    if isinstance(parsed, dict) and isinstance(parsed.get("passed"), bool):
        return parsed["passed"]
    if isinstance(parsed, bool):
        return parsed
    return None


def llm_judge(
    agent_reply: str,
    evaluation: Dict[str, Any],
    judge_client: Optional[LocalLLMClient] = None,
    judge_max_tokens: int = 256,
) -> bool:
    """
    Evaluate whether the agent's final reply resolves the ticket.

    Uses the configured local LLM judge when available. If no judge client
    is provided, or if the judge output cannot be parsed, falls back to the
    previous placeholder behavior of checking for a non-empty final reply.
    """
    if not agent_reply.strip():
        return False

    reference_admin_reply = evaluation.get("reference_admin_reply", "")
    final_solution_criteria = evaluation.get("final_solution_criteria", [])
    expected_trajectory = evaluation.get("expected_trajectory", [])
    has_reference_admin_reply = bool(evaluation.get("has_reference_admin_reply", False))

    if judge_client is None:
        return True

    criteria_text = "\n".join(
        f"- {criterion}" for criterion in final_solution_criteria
    ) or "- No explicit criteria provided."
    trajectory_text = "\n".join(
        f"- {step}" for step in expected_trajectory
    ) or "- No explicit trajectory provided."

    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Expected trajectory:\n"
                f"{trajectory_text}\n\n"
                "Final solution criteria:\n"
                f"{criteria_text}\n\n"
                "Reference admin reply availability:\n"
                f"{has_reference_admin_reply}\n\n"
                "Reference admin reply:\n"
                f"{reference_admin_reply if has_reference_admin_reply else '(not available for this sample)'}\n\n"
                "Agent final reply:\n"
                f"{agent_reply}"
            ),
        },
    ]
    raw_verdict = judge_client.generate(
        judge_messages,
        temperature=0.0,
        max_tokens=judge_max_tokens,
    )
    verdict = _parse_judge_verdict(raw_verdict)
    return bool(agent_reply.strip()) if verdict is None else verdict


# ------------------------------------------------------------------ #
#  Core evaluation function                                           #
# ------------------------------------------------------------------ #

def evaluate_agent(
    agent: BaseHPCAgent,
    dataset: List[Dict[str, Any]],
    max_steps: int = 15,
    judge_client: Optional[LocalLLMClient] = None,
    judge_max_tokens: int = 256,
    include_reconstructed: bool = False,
) -> Dict[str, Any]:
    """
    Run *agent* on every ticket in *dataset* and compute aggregate metrics.

    Args:
        agent:     An instance that implements :class:`BaseHPCAgent`.
        dataset:   List of ticket dicts in the cleaned benchmark schema.
        max_steps: Maximum interaction steps per ticket before giving up.
        judge_client:
                   Optional local LLM client used for task-success judging.
        judge_max_tokens:
                   Maximum generated tokens for each judge call.

    Returns:
        A dict with aggregate metrics including ``task_success_rate``,
        ``avg_tool_selection_recall``, ``avg_param_match_score``,
        compatibility alias ``avg_action_recall``, and
        ``avg_efficiency_penalty``.
    """
    valid_tickets = []
    invalid_skipped = 0
    reconstructed_skipped = 0
    grounded_count = 0
    reconstructed_count = 0

    for ticket in dataset:
        if ticket.get("is_valid", True) is False:
            invalid_skipped += 1
            continue
        release_tier = ticket.get("release_tier", "grounded")
        if release_tier == "grounded":
            grounded_count += 1
        elif release_tier == "reconstructed":
            reconstructed_count += 1
            if not include_reconstructed:
                reconstructed_skipped += 1
                continue
        valid_tickets.append(ticket)

    skipped_tickets = invalid_skipped + reconstructed_skipped
    total_tickets = len(valid_tickets)
    if total_tickets == 0:
        if skipped_tickets:
            print(
                "No valid tickets to evaluate. "
                f"Skipped {skipped_tickets} invalid ticket(s)."
            )
        else:
            print("Dataset is empty. Exiting.")
        return {}

    successes = 0
    total_tool_selection_recall_pct = 0.0
    total_param_match_score = 0.0
    total_errors = 0

    if skipped_tickets:
        if invalid_skipped:
            print(f"Skipping {invalid_skipped} invalid ticket(s) with is_valid=false.")
        if reconstructed_skipped:
            print(
                f"Skipping {reconstructed_skipped} reconstructed ticket(s); "
                "pass include_reconstructed=True to evaluate them."
            )
        print()

    print(f"Starting evaluation on {total_tickets} ticket(s)...\n")

    for i, ticket in enumerate(valid_tickets):
        ticket_id = ticket.get("instance_id", f"Unknown_{i}")
        print(f"--- Ticket {i + 1}/{total_tickets}: {ticket_id} ---")

        # Reset agent state for a fresh interaction
        agent.reset()
        if hasattr(agent, "begin_ticket"):
            agent.begin_ticket(ticket_id)

        # Initialise sandbox
        sandbox = HPCSandbox(ticket)

        # Build the initial observation the agent sees
        observation = ticket.get("instruction", "")

        done = False
        step_count = 0
        errors_this_ticket = 0
        matched_steps_this_ticket = 0
        param_score_sum_this_ticket = 0.0
        agent_reply = ""
        step_infos: List[Dict[str, Any]] = []

        # ---- Interaction loop (lives HERE, not in the agent) ----
        while not done and step_count < max_steps:
            action = agent.take_action(observation)
            observation, done, info = sandbox.step(action)
            step_infos.append(info)

            if not done and info.get("status") == "error" and not info.get("action_match", False):
                errors_this_ticket += 1

            if info.get("action_match"):
                matched_steps_this_ticket += 1
                param_score_sum_this_ticket += float(info.get("param_score", 0.0))

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
        evaluation = ticket.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        success = llm_judge(
            agent_reply,
            evaluation,
            judge_client=judge_client,
            judge_max_tokens=judge_max_tokens,
        )
        if success:
            successes += 1

        # 2. Tool-selection recall and parameter quality
        trace_count = len(sandbox.traces)
        if trace_count > 0:
            executed_count = sum(sandbox.executed_traces)
            tool_selection_recall_pct = (executed_count / trace_count) * 100.0
        else:
            tool_selection_recall_pct = 100.0

        if matched_steps_this_ticket > 0:
            param_match_score_pct = (param_score_sum_this_ticket / matched_steps_this_ticket) * 100.0
        else:
            param_match_score_pct = 0.0

        total_tool_selection_recall_pct += tool_selection_recall_pct
        total_param_match_score += param_match_score_pct

        # 3. Efficiency Penalty
        total_errors += errors_this_ticket

        print(f"  Steps:          {step_count}")
        print(f"  Invalid/Errors: {errors_this_ticket}")
        print(f"  Tool Recall:    {tool_selection_recall_pct:.2f}%")
        print(f"  Param Score:    {param_match_score_pct:.2f}%")
        print(f"  Task Success:   {success}\n")

        if hasattr(agent, "get_debug_transcript"):
            debug_transcript = agent.get_debug_transcript()
            if debug_transcript:
                print("  Agent Debug Transcript:")
                print(pformat(debug_transcript, width=100, compact=False))
                print()
                print("  Environment Match Info:")
                print(pformat(step_infos, width=100, compact=False))
                print()

    # ---- Aggregate metrics ----
    avg_success_rate = (successes / total_tickets) * 100.0
    avg_tool_selection_recall = total_tool_selection_recall_pct / total_tickets
    avg_param_match_score = total_param_match_score / total_tickets
    avg_efficiency_penalty = total_errors / total_tickets

    print("=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total Tickets Evaluated:  {total_tickets}")
    print(f"Grounded Tickets Seen:    {grounded_count}")
    print(f"Reconstructed Seen:      {reconstructed_count}")
    if invalid_skipped:
        print(f"Invalid Tickets Skipped: {invalid_skipped}")
    if reconstructed_skipped:
        print(f"Reconstructed Skipped:   {reconstructed_skipped}")
    print(f"Task Success Rate:        {avg_success_rate:.2f}%")
    print(f"Average Tool Recall:      {avg_tool_selection_recall:.2f}%")
    print(f"Average Param Score:      {avg_param_match_score:.2f}%")
    print(f"Efficiency Penalty:       {avg_efficiency_penalty:.2f} errors/ticket")

    return {
        "task_success_rate": avg_success_rate,
        "avg_tool_selection_recall": avg_tool_selection_recall,
        "avg_param_match_score": avg_param_match_score,
        "avg_action_recall": avg_tool_selection_recall,
        "avg_efficiency_penalty": avg_efficiency_penalty,
        "grounded_count": grounded_count,
        "reconstructed_count": reconstructed_count,
        "invalid_count": invalid_skipped,
        "reconstructed_skipped": reconstructed_skipped,
    }


# ------------------------------------------------------------------ #
#  Inline smoke test agent + fixture                                   #
# ------------------------------------------------------------------ #

class _SmokeTestAgent(BaseHPCAgent):
    """Deterministic agent used only for the local smoke test."""

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def take_action(self, observation: str) -> Dict[str, Any]:
        self._step += 1
        if self._step == 1:
            return {
                "action": "execute_bash",
                "params": {"command": "sacct -j 4242 --format=State,ExitCode"},
            }
        if self._step == 2:
            return {
                "action": "search_docs",
                "params": {"query": "Slurm exit code 137 out of memory"},
            }
        return {
            "action": "reply_user",
            "params": {
                "text": "The job hit an out-of-memory condition. Request more "
                "memory or reduce memory usage before rerunning."
            },
        }


if __name__ == "__main__":
    dummy_dataset = [
        {
            "instance_id": "DEMO_001",
            "is_valid": True,
            "release_tier": "grounded",
            "construction_mode": "extracted",
            "instruction": "My simulation exits immediately and the Slurm job "
            "shows exit code 137. Please help me figure out why.",
            "traces": [
                {
                    "trigger_command": (
                        "execute_bash(sacct -j 4242 --format=State,ExitCode)"
                    ),
                    "mock_output": "State|ExitCode\nOUT_OF_MEMORY|137:0",
                },
                {
                    "trigger_command": (
                        "search_docs(Slurm exit code 137 out of memory)"
                    ),
                    "mock_output": (
                        "Exit code 137 usually indicates the kernel killed "
                        "the process after it exceeded its memory allocation."
                    ),
                },
            ],
            "evaluation": {
                "expected_trajectory": [
                    "Check the job accounting record",
                    "Confirm the meaning of exit code 137 in the docs",
                ],
                "final_solution_criteria": [
                    "Identify the out-of-memory failure",
                    "Advise the user to request more memory or reduce usage",
                ],
                "has_reference_admin_reply": True,
                "reference_admin_reply": (
                    "Your job was terminated after exceeding its memory "
                    "allocation. Please rerun with more memory."
                ),
            },
        },
        {
            "instance_id": "DEMO_INVALID",
            "is_valid": False,
            "release_tier": "reconstructed",
            "construction_mode": "reconstructed",
            "instruction": "This placeholder ticket should be skipped.",
            "traces": [],
            "evaluation": {
                "expected_trajectory": [],
                "final_solution_criteria": [],
                "has_reference_admin_reply": False,
                "reference_admin_reply": "",
            },
        }
    ]

    print("Running integration test with _SmokeTestAgent...\n")
    evaluate_agent(_SmokeTestAgent(), dummy_dataset)
