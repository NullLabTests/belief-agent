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


Message = dict[str, str]


class LLMClient:
    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        raise NotImplementedError

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        raise NotImplementedError


class LiteLLMClient(LLMClient):
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
    def __init__(
        self,
        client: LLMClient | None = None,
        state: BeliefState | None = None,
        system_prompt_style: str = "bullet",
        extra_instructions: str = "",
        auto_reflect: bool = False,
        reflect_mode: str = "auto",
        reflect_callback: Callable[[list[dict]], bool] | None = None,
        name: str = "agent",
        llm_call: Callable[[list[Message]], str] | None = None,
    ) -> None:
        self.client = client or (LiteLLMClient() if HAS_LITELLM else None)
        self._llm_call = llm_call
        self.state = state or BeliefState()
        self.system_prompt_style = system_prompt_style
        self.extra_instructions = extra_instructions
        self.auto_reflect = auto_reflect
        self.reflect_mode = reflect_mode
        self.reflect_callback = reflect_callback
        self.name = name
        self._messages: list[dict] = []

    # ---- chat ----

    def chat(self, message: str, **kwargs: Any) -> str:
        self._rebuild_system_prompt()
        self._messages.append({"role": "user", "content": message})
        response = self._generate(**kwargs)
        self._messages.append({"role": "assistant", "content": response})
        self._update_beliefs_from(response)
        if self.auto_reflect:
            self._run_reflection(message, response)
        return response

    def chat_stream(self, message: str, **kwargs: Any) -> Generator[str, None, None]:
        self._rebuild_system_prompt()
        self._messages.append({"role": "user", "content": message})
        collected: list[str] = []
        for token in self._generate_stream(**kwargs):
            collected.append(token)
            yield token
        full = "".join(collected)
        self._messages.append({"role": "assistant", "content": full})
        self._update_beliefs_from(full)
        if self.auto_reflect:
            self._run_reflection(message, full)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        system_prompt = build_system_prompt(
            self.state,
            style=self.system_prompt_style,
            extra_instructions=self.extra_instructions,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages, **kwargs)

    def __call__(self, user_message: str) -> str:
        return self.chat(user_message)

    # ---- generation ----

    def _generate(self, **kwargs: Any) -> str:
        if self._llm_call:
            return self._llm_call(self._messages)
        if self.client:
            return self.client.complete(self._messages, **kwargs)
        raise ValueError("No LLM client or llm_call provided")

    def _generate_stream(self, **kwargs: Any) -> Generator[str, None, None]:
        if self.client:
            yield from self.client.complete_stream(self._messages, **kwargs)
        else:
            yield self._generate(**kwargs)

    def _call_llm(self, messages: list[dict], **kwargs: Any) -> str:
        if self._llm_call:
            return self._llm_call(messages)
        if self.client:
            return self.client.complete(messages, **kwargs)
        raise ValueError("No LLM client or llm_call provided")

    # ---- belief injection ----

    def _rebuild_system_prompt(self) -> None:
        prompt = build_system_prompt(
            self.state,
            style=self.system_prompt_style,
            extra_instructions=self.extra_instructions,
        )
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = prompt
        else:
            self._messages.insert(0, {"role": "system", "content": prompt})

    def _belief_context(self) -> str:
        if not self.state.beliefs:
            return ""
        lines = ["<beliefs>"]
        for b in self.state.beliefs:
            status = "confident" if b.confidence >= 0.7 else "uncertain"
            if b.contradictions:
                status = "contradicted"
            lines.append(
                f'  - "{b.fact}" confidence={b.confidence:.2f} [{status}]'
            )
        lines.append("</beliefs>")
        return "\n".join(lines)

    def _update_beliefs_from(self, reply: str) -> None:
        raw = reply.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                data = [data]
            for item in data:
                if "fact" in item or "proposition" in item:
                    fact = item.get("fact") or item.get("proposition", "")
                    if fact and not self.state.query(fact):
                        self.state.add_belief(
                            fact=fact,
                            confidence=item.get("confidence", 0.5),
                            source=item.get("source", "model"),
                            evidence=item.get("evidence", []),
                            tags=["extracted"],
                            detect_contradictions=False,
                        )
        except (json.JSONDecodeError, KeyError):
            pass

    # ---- reflection ----

    def _run_reflection(self, user_message: str, response: str) -> None:
        reflect(
            state=self.state,
            user_message=user_message,
            response=response,
            mode=self.reflect_mode,
            callback=self.reflect_callback,
            client=self._get_client_for_reflection(),
        )

    def _get_client_for_reflection(self):
        class _Adapter:
            def __init__(self, agent):
                self._agent = agent
            def complete(self, messages, **kwargs):
                return self._agent._call_llm(messages, **kwargs)
        return _Adapter(self)

    def reflect(self) -> str:
        msg = (
            "Reflect on your current beliefs. For each belief, state whether "
            "you still hold it, whether you have found contradictions, and "
            "what your updated confidence level is. "
        )
        if self.state.beliefs:
            msg += f"Current beliefs:\n{self._belief_context()}\n"
        msg += (
            'Return as JSON: '
            '[{"fact":"...","confidence":0.X,"contradictions":[...],"evidence":[...]}]'
        )
        return self.complete(msg)

    # ---- belief management ----

    def adopt(self, fact: str, confidence: float = 0.5, source: str = "user",
              evidence: list[str] | None = None, tags: list[str] | None = None) -> None:
        matches = self.state.query(fact)
        if matches:
            b = matches[0]
            b.confidence = (b.confidence + confidence) / 2.0
            if evidence:
                for e in evidence:
                    if e not in b.evidence:
                        b.evidence.append(e)
            b.timestamp = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
            self.state.version += 1
        else:
            self.state.add_belief(
                fact=fact, confidence=confidence, source=source,
                evidence=evidence, tags=tags,
                detect_contradictions=True,
            )

    def get_belief(self, fact: str) -> Any | None:
        matches = self.state.query(fact)
        return matches[0] if matches else None

    def update_belief(self, fact: str, confidence: float | None = None,
                      evidence: list[str] | None = None) -> bool:
        matches = self.state.query(fact)
        if not matches:
            return False
        b = matches[0]
        if confidence is not None:
            b.confidence = confidence
        if evidence is not None:
            b.evidence = evidence
        self.state.version += 1
        return True

    def remove_belief(self, fact: str) -> bool:
        matches = self.state.query(fact)
        if not matches:
            return False
        for b in matches:
            self.state.remove_belief(b.id)
        return True

    def list_beliefs(self, min_confidence: float = 0.0,
                     only_contradicted: bool = False) -> list[Any]:
        return self.state.list_beliefs(
            min_confidence=min_confidence, only_contradicted=only_contradicted
        )

    # ---- history ----

    @property
    def message_history(self) -> list[dict]:
        return list(self._messages)

    def reset_conversation(self, keep_beliefs: bool = True) -> None:
        self._messages.clear()
        if not keep_beliefs:
            self.state = BeliefState()

    def reset_history(self) -> None:
        self._messages.clear()

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt_style": self.system_prompt_style,
            "extra_instructions": self.extra_instructions,
            "state": self.state.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
