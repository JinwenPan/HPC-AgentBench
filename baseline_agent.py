"""
Naive local-LLM baseline agent for HPC-AgentBench.

Uses either a local vLLM OpenAI-compatible server or in-process vLLM
offline inference to decide actions in a multi-turn loop driven by the
evaluator. Conversation history is maintained across steps within a
single ticket and cleared on ``reset()``.
"""

import json
from pprint import pformat
from typing import Any, Dict, List, Optional

from interface import BaseHPCAgent
from llm_config import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OFFLINE_DTYPE,
    DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
    DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
    DEFAULT_OFFLINE_MAX_MODEL_LEN,
    DEFAULT_OFFLINE_QUANTIZATION,
    DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
    DEFAULT_SERVER_API_KEY,
    DEFAULT_SERVER_BASE_URL,
    DEFAULT_TEMPERATURE,
)
from local_llm import LocalLLMClient

SYSTEM_PROMPT = """\
You are an expert HPC (High-Performance Computing) support agent.
You diagnose and resolve user issues by invoking diagnostic tools.

On every turn you receive an observation (either the ticket instruction
or the output of the last tool you called). You MUST respond with
**exactly one** JSON object — no markdown fences, no commentary — in one
of the two forms below:

1. Invoke a tool:
   {"action": "execute_bash", "params": {"command": "<bash command>"}}
   {"action": "search_docs", "params": {"query": "<search query>"}}
   {"action": "ask_user_for_info", "params": {"question": "<question>"}}

2. Finish and reply to the user:
   {"action": "reply_user", "params": {"text": "Your final resolution."}}

Do not invent any other tool names or parameter keys.
Use at most one tool per turn.
Think step-by-step. Gather evidence before concluding. When you have
enough evidence, reply to the user with a clear resolution.
"""


class NaiveLocalLLMAgent(BaseHPCAgent):
    """
    Baseline agent that delegates every decision to a local LLM backend.

    Args:
        model:       Model identifier (default ``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8``).
        backend:     ``"server"`` for a local OpenAI-compatible vLLM server or
                     ``"offline"`` for in-process vLLM inference.
        temperature: Sampling temperature (default ``0.0`` for determinism).
        max_tokens:  Maximum generated tokens per agent turn.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        backend: str = DEFAULT_LLM_BACKEND,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        server_base_url: str = DEFAULT_SERVER_BASE_URL,
        server_api_key: str = DEFAULT_SERVER_API_KEY,
        offline_tensor_parallel_size: int = DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
        offline_gpu_memory_utilization: float = DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
        offline_max_model_len: int = DEFAULT_OFFLINE_MAX_MODEL_LEN,
        offline_dtype: str = DEFAULT_OFFLINE_DTYPE,
        offline_quantization: str = DEFAULT_OFFLINE_QUANTIZATION,
        offline_enable_chunked_prefill: bool = DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
        verbose: bool = False,
        client: Optional[LocalLLMClient] = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._owns_client = client is None
        self._current_ticket_id = "unknown"
        self.client = client or LocalLLMClient(
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            server_base_url=server_base_url,
            server_api_key=server_api_key,
            offline_tensor_parallel_size=offline_tensor_parallel_size,
            offline_gpu_memory_utilization=offline_gpu_memory_utilization,
            offline_max_model_len=offline_max_model_len,
            offline_dtype=offline_dtype,
            offline_quantization=offline_quantization,
            offline_enable_chunked_prefill=offline_enable_chunked_prefill,
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.debug_events: List[Dict[str, Any]] = []
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
        return self.client.generate(
            self.conversation_history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

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
            if not isinstance(parsed["params"], dict):
                raise ValueError("'params' must be a JSON object.")
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
        self.debug_events = []

    def begin_ticket(self, ticket_id: str) -> None:
        """Record the current ticket ID for verbose debug output."""
        self._current_ticket_id = ticket_id

    def get_debug_transcript(self) -> List[Dict[str, Any]]:
        """Return a copy of the per-step debug events collected so far."""
        return list(self.debug_events)

    def _debug_print(self, label: str, payload: Any) -> None:
        if not self.verbose:
            return
        print(f"[agent-debug][ticket={self._current_ticket_id}] {label}:")
        if isinstance(payload, str):
            print(payload)
        else:
            print(pformat(payload, width=100, compact=False))
        print()

    def close(self) -> None:
        """Close the owned inference client, if any."""
        if self._owns_client:
            self.client.close()

    def take_action(self, observation: str) -> Dict[str, Any]:
        """
        Append the latest observation, query the LLM, and return an action.
        """
        step_index = len(self.debug_events) + 1
        self._debug_print(f"step {step_index} observation", observation)
        self.conversation_history.append(
            {"role": "user", "content": observation}
        )

        raw_response = self._call_llm()
        self._debug_print(f"step {step_index} raw_response", raw_response)

        # Record the assistant's raw output in history
        self.conversation_history.append(
            {"role": "assistant", "content": raw_response}
        )

        parsed_action = self._parse_action(raw_response)
        self._debug_print(f"step {step_index} parsed_action", parsed_action)
        self.debug_events.append(
            {
                "step": step_index,
                "observation": observation,
                "raw_response": raw_response,
                "parsed_action": parsed_action,
            }
        )

        return parsed_action


NaiveOpenAIAgent = NaiveLocalLLMAgent
