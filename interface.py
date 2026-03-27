from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseHPCAgent(ABC):
    """
    Abstract base class defining the contract for all HPC benchmark agents.

    Every agent must implement `take_action`, which receives a textual
    observation (the ticket instruction or a tool output) and returns a
    structured action dictionary.

    Action schema
    -------------
    The benchmark exposes exactly three tools, each with a single string
    parameter:

        {"action": "execute_bash", "params": {"command": "..."}}
        {"action": "search_docs", "params": {"query": "..."}}
        {"action": "ask_user_for_info", "params": {"question": "..."}}

    To finish and reply to the user:
        {"action": "reply_user", "params": {"text": "final resolution..."}}
    """

    def reset(self) -> None:
        """Reset internal state (e.g. conversation history) for a new ticket."""
        pass

    @abstractmethod
    def take_action(self, observation: str) -> Dict[str, Any]:
        """
        Decide the next action given the current observation.

        Args:
            observation: The initial ticket instruction or the string returned by the
                         previous environment step.

        Returns:
            A dict with keys ``"action"`` and ``"params"`` following the
            action schema described in the class docstring.
        """
        ...
