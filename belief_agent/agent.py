"""Agent wrapper that injects belief state into LLM conversations."""

from __future__ import annotations

import json
from typing import Any, Callable, Generator

from .belief_state import BeliefState
from .reflection import reflect
from .utils import build_system_prompt

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class LLMClient:
    """Abstract interface for an LLM provider.

    Subclass this to support models not covered by LiteLLM.
    """

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """Send messages and return the text response."""
        raise NotImplementedError

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        """Send messages and stream the response."""
        raise NotImplementedError


class LiteLLMClient(LLMClient):
    """Adapter that wraps LiteLLM's ``completion`` function.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``,
        ``"ollama/llama3"``, …).
    **kwargs:
        Default parameters passed to every call (temperature, max_tokens, …).
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        if not HAS_LITELLM:
            raise ImportError(
                "LiteLLM is required. Install it with: pip install 'belief-agent[litellm]'"
            )
        self.model = model
        self.default_kwargs = kwargs

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        merged = {**self.default_kwargs, **kwargs}
        response = litellm.completion(model=self.model, messages=messages, **merged)
        return response.choices[0].message.content

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        merged = {**self.default_kwargs, **kwargs}
        response = litellm.completion(model=self.model, messages=messages, stream=True, **merged)
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class BeliefAgent:
    """An LLM agent that maintains an explicit :class:`BeliefState` and
    automatically injects it into every conversation.

    Parameters
    ----------
    client:
        An LLM client instance.  Defaults to ``LiteLLMClient("gpt-4o-mini")``.
    state:
        An existing belief state, or a fresh one is created.
    system_prompt_style:
        How beliefs are formatted in the system prompt (``"bullet"``,
        ``"json"``, ``"natural"``).
    extra_instructions:
        Extra instructions appended to the system prompt.
    auto_reflect:
        If True, run :func:`~belief_agent.reflection.reflect` after every
        LLM response.
    reflect_mode:
        ``"auto"`` — apply all proposed updates automatically;
        ``"human"`` — ask for confirmation via a callback.
    reflect_callback:
        Function called with proposed updates for human-in-the-loop review.
        Signature: ``callback(updates: list[dict]) -> bool``.
        Return ``True`` to apply, ``False`` to skip.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        state: BeliefState | None = None,
        system_prompt_style: str = "bullet",
        extra_instructions: str = "",
        auto_reflect: bool = False,
        reflect_mode: str = "auto",
        reflect_callback: Callable[[list[dict]], bool] | None = None,
    ) -> None:
        self.client = client or (LiteLLMClient() if HAS_LITELLM else _raise_no_client())
        self.state = state or BeliefState()
        self.system_prompt_style = system_prompt_style
        self.extra_instructions = extra_instructions
        self.auto_reflect = auto_reflect
        self.reflect_mode = reflect_mode
        self.reflect_callback = reflect_callback
        self._messages: list[dict] = []

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, message: str, **kwargs: Any) -> str:
        """Send a user message and get a response.

        The system prompt is rebuilt before each call to include the latest
        belief state.

        Parameters
        ----------
        message:
            The user's message.
        **kwargs:
            Overrides for LLM call parameters.

        Returns
        -------
        The model's text response.
        """
        self._rebuild_system_prompt()
        self._messages.append({"role": "user", "content": message})
        response = self.client.complete(self._messages, **kwargs)
        self._messages.append({"role": "assistant", "content": response})
        if self.auto_reflect:
            self._run_reflection(message, response)
        return response

    def chat_stream(self, message: str, **kwargs: Any) -> Generator[str, None, None]:
        """Like :meth:`chat` but streams tokens."""
        self._rebuild_system_prompt()
        self._messages.append({"role": "user", "content": message})
        collected: list[str] = []
        for token in self.client.complete_stream(self._messages, **kwargs):
            collected.append(token)
            yield token
        full = "".join(collected)
        self._messages.append({"role": "assistant", "content": full})
        if self.auto_reflect:
            self._run_reflection(message, full)

    # ------------------------------------------------------------------
    # Low-level completion (stateless)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Stateless completion — injects beliefs but does **not** maintain
        conversation history.

        Useful for one-off calls or tool calls.
        """
        system_prompt = build_system_prompt(
            self.state,
            style=self.system_prompt_style,
            extra_instructions=self.extra_instructions,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.client.complete(messages, **kwargs)

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def _run_reflection(self, user_message: str, response: str) -> None:
        reflect(
            state=self.state,
            user_message=user_message,
            response=response,
            mode=self.reflect_mode,
            callback=self.reflect_callback,
            client=self.client,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_system_prompt(self) -> None:
        prompt = build_system_prompt(
            self.state,
            style=self.system_prompt_style,
            extra_instructions=self.extra_instructions,
        )
        # Replace or insert system message at position 0
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = prompt
        else:
            self._messages.insert(0, {"role": "system", "content": prompt})

    @property
    def message_history(self) -> list[dict]:
        """Read-only access to the conversation history."""
        return list(self._messages)

    def reset_conversation(self, keep_beliefs: bool = True) -> None:
        """Clear the message history.

        Parameters
        ----------
        keep_beliefs:
            If True, the belief state is preserved.
        """
        self._messages.clear()
        if not keep_beliefs:
            self.state = BeliefState()


def _raise_no_client() -> LLMClient:
    raise ValueError(
        "No LLM client provided and LiteLLM is not installed. "
        "Install with: pip install 'belief-agent[litellm]'"
    )
