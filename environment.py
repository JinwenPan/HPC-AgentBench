from typing import Any, Dict, List, Tuple

from benchmark_semantics import (
    extract_action_invocation,
    normalize_trigger_command,
    param_match_threshold,
    parse_trigger_command,
    score_tool_arguments,
)


class HPCSandbox:
    """
    Offline, trace-based sandbox that simulates HPC tool responses.

    Initialised with a single ticket's JSON data, which contains:

    * ``instruction`` – the first observation shown to the agent
    * ``traces`` – an order-agnostic list of tool invocations to match

    Each trace item contains:

    * ``trigger_command`` – canonical tool call in ``action(argument)`` form
    * ``mock_output`` – the canned response returned on a match

    The sandbox is **order-agnostic**: traces can be triggered in
    any sequence.  Each action can only be "consumed" once.
    """

    def __init__(self, ticket_data: Dict[str, Any]) -> None:
        self.ticket_data = ticket_data
        self.instruction = ticket_data.get("instruction", "")
        traces_raw = ticket_data.get("traces", [])
        self.traces: List[Dict[str, Any]] = []

        if isinstance(traces_raw, list):
            for trace in traces_raw:
                if not isinstance(trace, dict):
                    continue

                trigger_command = trace.get("trigger_command")
                mock_output = trace.get("mock_output")
                if not isinstance(trigger_command, str) or not isinstance(
                    mock_output, str
                ):
                    continue

                parsed_trigger = parse_trigger_command(trigger_command)
                if parsed_trigger is None:
                    continue
                action_name, argument, normalized_trigger = parsed_trigger

                self.traces.append(
                    {
                        "action": action_name,
                        "argument": argument,
                        "trigger_command": normalized_trigger,
                        "mock_output": mock_output,
                    }
                )

        self.executed_traces: List[bool] = [False] * len(self.traces)

    def step(
        self, action_dict: Dict[str, Any]
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Process one agent action against the trace data.

        Args:
            action_dict: ``{"action": "<name>", "params": {...}}``

        Returns:
            A 3-tuple ``(observation, is_done, info)``:

            * *observation* – textual result for the agent.
            * *is_done*     – ``True`` when ``reply_user`` is called.
            * *info*        – metadata about what happened (e.g. match status).
        """
        action_name = action_dict.get("action") if isinstance(action_dict, dict) else None
        params = action_dict.get("params", {}) if isinstance(action_dict, dict) else {}

        # --- Terminal action: agent replies to the user ---
        if action_name == "reply_user":
            reply_text = params.get("text", "") if isinstance(params, dict) else ""
            return reply_text, True, {"status": "reply"}

        invocation = extract_action_invocation(action_dict)
        if invocation is None:
            return (
                "Observation: Error - Invalid action or parameters.",
                False,
                {
                    "status": "error",
                    "action_match": False,
                    "param_match": False,
                    "param_score": 0.0,
                    "teacher_forced": False,
                    "matched_trace": None,
                    "expected_trigger_command": None,
                },
            )
        action_name, argument, normalized_trigger, schema_valid = invocation

        best_index = None
        best_trace = None
        best_score = -1.0

        # --- Order-agnostic matching against traces ---
        for idx, trace in enumerate(self.traces):
            if self.executed_traces[idx]:
                continue  # already consumed

            if trace.get("action") != action_name:
                continue

            score = score_tool_arguments(
                action_name,
                argument,
                trace.get("argument", ""),
            )
            if score > best_score:
                best_index = idx
                best_trace = trace
                best_score = score

        if best_trace is not None and best_index is not None:
            self.executed_traces[best_index] = True
            threshold = param_match_threshold(action_name)
            param_match = best_score >= threshold and schema_valid
            observation = best_trace.get("mock_output", "")
            return observation, False, {
                "status": "match",
                "action_match": True,
                "param_match": param_match,
                "param_score": max(best_score, 0.0),
                "teacher_forced": not param_match,
                "matched_trace": best_trace,
                "expected_trigger_command": best_trace.get("trigger_command"),
                "trigger_command": normalized_trigger,
                "schema_valid": schema_valid,
            }

        # --- No match found ---
        return (
            "Observation: Error - Invalid action or parameters.",
            False,
            {
                "status": "error",
                "action_match": False,
                "param_match": False,
                "param_score": 0.0,
                "teacher_forced": False,
                "matched_trace": None,
                "expected_trigger_command": None,
                "trigger_command": normalized_trigger,
                "schema_valid": schema_valid,
            },
        )
