from typing import Dict, Any, Tuple, List


class HPCSandbox:
    """
    Offline, trace-based sandbox that simulates HPC tool responses.

    Initialised with a single ticket's JSON data, which contains a list of
    ``required_actions``.  Each required action specifies:

    * ``expected_action``  – the tool name the agent should call
    * ``expected_params``  – the parameters the agent should supply
    * ``mock_observation`` – the canned response returned on a match

    The sandbox is **order-agnostic**: required actions can be triggered in
    any sequence.  Each action can only be "consumed" once.
    """

    def __init__(self, ticket_data: Dict[str, Any]) -> None:
        self.ticket_data = ticket_data
        self.required_actions: List[Dict[str, Any]] = ticket_data.get(
            "required_actions", []
        )
        self.executed_actions: List[bool] = [False] * len(self.required_actions)

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
        action_name = action_dict.get("action")
        params = action_dict.get("params", {})

        # --- Terminal action: agent replies to the user ---
        if action_name == "reply_user":
            return params.get("text", ""), True, {}

        # --- Order-agnostic matching against required actions ---
        for idx, req in enumerate(self.required_actions):
            if self.executed_actions[idx]:
                continue  # already consumed

            if (
                req.get("expected_action") == action_name
                and req.get("expected_params") == params
            ):
                self.executed_actions[idx] = True
                observation = req.get("mock_observation", "")
                return observation, False, {
                    "status": "match",
                    "matched_action": req,
                }

        # --- No match found ---
        return (
            "Observation: Error - Invalid action or parameters.",
            False,
            {"status": "error"},
        )
