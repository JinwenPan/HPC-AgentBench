"""
Naive OpenAI Baseline Agent for HPC-AgentBench.

Uses the OpenAI chat completions API to decide actions in a multi-turn
loop driven by the evaluator.  Conversation history is maintained across
steps within a single ticket and cleared on ``reset()``.
"""

import json
import os
from typing import Dict, Any, List

from openai import OpenAI

from interface import BaseHPCAgent

SYSTEM_PROMPT = """\
You are an expert HPC (High-Performance Computing) support agent.
You diagnose and resolve user issues by invoking diagnostic tools.

On every turn you receive an observation (either the user's initial query
or the output of the last tool you called).  You MUST respond with
**exactly one** JSON object — no markdown fences, no commentary — in one
of the two forms below:

1. Invoke a tool:
   {"action": "<tool_name>", "params": {"key": "value"}}

2. Finish and reply to the user:
   {"action": "reply_user", "params": {"text": "Your final resolution."}}

Common tools include (but are not limited to):
  get_job_state, check_syslog, check_gpu_status, module_avail,
  check_disk_quota, run_command.

Think step-by-step. Gather information before concluding. When you have
enough evidence, reply to the user with a clear resolution.
"""


class NaiveOpenAIAgent(BaseHPCAgent):
    """
    Baseline agent that delegates every decision to an OpenAI chat model.

    Args:
        model:       Model identifier (default ``"gpt-4o-mini"``).
        temperature: Sampling temperature (default ``0.0`` for determinism).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.conversation_history: List[Dict[str, str]] = []
        self._init_history()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _init_history(self) -> None:
        """Set up conversation history with the system prompt."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def _call_llm(self) -> str:
        """Send the current conversation to the model and return its text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_action(text: str) -> Dict[str, Any]:
        """
        Attempt to parse a JSON action dict from the model's raw output.

        Falls back to a ``reply_user`` action echoing the raw text if
        parsing fails.
        """
        # Strip markdown code fences if the model wraps its output
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            # Minimal validation
            if "action" not in parsed:
                raise ValueError("Missing 'action' key.")
            if "params" not in parsed:
                parsed["params"] = {}
            return parsed
        except (json.JSONDecodeError, ValueError):
            # Graceful fallback: treat malformed output as a final reply
            return {
                "action": "reply_user",
                "params": {"text": text},
            }

    # ------------------------------------------------------------------ #
    #  BaseHPCAgent interface                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear conversation history for a new ticket."""
        self._init_history()

    def take_action(self, observation: str) -> Dict[str, Any]:
        """
        Append the latest observation, query the LLM, and return an action.
        """
        self.conversation_history.append(
            {"role": "user", "content": observation}
        )

        raw_response = self._call_llm()

        # Record the assistant's raw output in history
        self.conversation_history.append(
            {"role": "assistant", "content": raw_response}
        )

        return self._parse_action(raw_response)
