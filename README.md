# belief-agent

**Explicit belief states, structured reflection, and human negotiation for LLM agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-78%25-brightgreen)](https://github.com/nulllabtests/belief-agent)

---

## The Problem

Current frontier models are **statistical pattern matchers**, not grounded reasoners. They:

- ❌ **Drift** across long conversations, contradicting their own earlier statements.
- ❌ **Hide uncertainty** — fluent text makes you think they're sure when they're guessing.
- ❌ **Cannot negotiate** conflicting objectives — they commit to one goal and ignore trade-offs.
- ❌ Have **no explicit beliefs** — everything is latent in the weights, making steerability and inspection impossible.

> *"We have agents that can write code, but we don't have agents that know what they believe."*

## The Solution

`belief-agent` adds three things on top of any LLM:

| Component | What it does |
|---|---|
| **BeliefState** | Stores structured, inspectable beliefs with confidence, source, timestamp, and contradiction tracking. |
| **Reflection** | After every response, analyzes output against current beliefs and proposes updates — automatically or with human approval. |
| **Negotiation** | Given conflicting goals, produces ranked priorities, feasibility scores, and concrete compromises. |

The result: **steerable, inspectable agents** that update their beliefs when reality contradicts them, and handle human objectives with nuance.

---

## Quick Install

```bash
pip install belief-agent
```

*Minimal deps: only `pydantic` is required. LiteLLM support is optional.*

---

## 30-Second Demo

```python
from belief_agent import BeliefState

# ---- 1. Beliefs ----
state = BeliefState()
state.add_belief("The sky is blue", confidence=0.9, source="observation")
state.add_belief("The sky is not blue", confidence=0.3, source="user")

# Auto-detects contradictions
for a, b in state.get_contradictions():
    print(f"Contradiction: {a.fact}  <->  {b.fact}")

# ---- 2. Query ----
for b in state.query("sky"):
    print(b)     # [90%] The sky is blue (source: observation)

# ---- 3. Update ----
b = state.query("sky is blue")[0]
state.update_belief(b.id, confidence=1.0)

# ---- 4. JSON Round-trip ----
data = state.serialize(indent=2)
restored = BeliefState.deserialize(data)
```

Output:

```
Contradiction: The sky is blue  <->  The sky is not blue
[90%] The sky is blue (source: observation)
[30%] The sky is not blue (source: user)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Application                     │
├─────────────────────────────────────────────────────────┤
│                     BeliefAgent                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ BeliefState  │  │  reflect()   │  │negotiate_    │  │
│  │              │  │              │  │goals()       │  │
│  │ • facts      │  │ • detect     │  │ • conflict   │  │
│  │ • confidence │  │   drift      │  │   detection  │  │
│  │ • source     │  │ • propose    │  │ • ranking    │  │
│  │ • timestamp  │  │   updates    │  │ • compromise │  │
│  │ • contra-    │  │ • auto or    │  │ • beliefs    │  │
│  │   dictions   │  │   human      │  │   extraction │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                            │                            │
├────────────────────────────┼────────────────────────────┤
│                     LLM Client (any provider)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │  OpenAI  │ │  Claude  │ │  Grok    │ │ Ollama/etc │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: Beliefs live **outside** the model — they are stored as structured data, injected into prompts, and updated explicitly. This makes the agent's "mind" transparent and debuggable.

---

## Full Examples

| Example | File | Description |
|---|---|---|
| **Basic Agent** | [`examples/basic_agent.py`](examples/basic_agent.py) | Belief CRUD, contradiction detection, merge, serialization |
| **Long-Horizon Planning** | [`examples/long_horizon_planning.py`](examples/long_horizon_planning.py) | Vanilla vs belief-agent: how reflection prevents drift |
| **Multi-Objective Negotiation** | [`examples/multi_objective_negotiation.py`](examples/multi_objective_negotiation.py) | Conflicts, ranking, compromises, belief extraction |

Run any example:

```bash
cd belief-agent
python3 examples/basic_agent.py
```

---

## Notebooks

Explore the concepts interactively:

| Notebook | Topic |
|---|---|
| [`01_quickstart.md`](notebooks/01_quickstart.md) | Create, query, update, merge, serialize beliefs |
| [`02_long_horizon.md`](notebooks/02_long_horizon.md) | Why vanilla agents drift and how reflection fixes it |
| [`03_negotiation.md`](notebooks/03_negotiation.md) | Multi-objective trade-off analysis with structured output |

---

## API Overview

### `BeliefState`

```python
state = BeliefState()
state.add_belief(fact, confidence, source, tags)
state.update_belief(id, fact=..., confidence=...)
state.remove_belief(id)
state.query(text)             # substring search
state.query_by_tag(tag)
state.query_by_source(source)
state.get_contradictions()    # returns list of (Belief, Belief) pairs
state.merge(other_state)      # merge with conflict resolution
state.serialize(indent=2)     # → JSON string
BeliefState.deserialize(json_str)
```

### `BeliefAgent`

```python
agent = BeliefAgent(
    client=LiteLLMClient("gpt-4o"),
    auto_reflect=True,          # run reflection after every response
    reflect_mode="auto",         # or "human" with callback
)

resp = agent.chat("Hello!")     # maintains conversation history
resp = agent.complete("Hello!") # stateless (no history)
for token in agent.chat_stream("Tell me a story"):
    print(token, end="")
```

### `reflect()`

```python
from belief_agent import reflect

updates = reflect(
    state=state,
    user_message="The sky is green",
    response="You're right, it's green.",
    client=client,
    mode="auto",       # or "human" with callback
)
```

### `negotiate_goals()`

```python
from belief_agent import negotiate_goals

result = negotiate_goals([
    {"goal": "Minimize cost",         "importance": 9},
    {"goal": "Maximize safety",       "importance": 10},
    {"goal": "Ship by end of month",  "importance": 7},
], client)

print(result["compromise"])
print(result["ranked_goals"])
print(result["recommended_beliefs"])   # ready to add to BeliefState
```

---

## Why This Matters

### Short-term

- **Debug agent behavior** — see exactly what your agent "believes" at any point.
- **Prevent drift** — reflection catches contradictions before they compound.
- **Handle human values** — negotiation makes trade-offs explicit, not latent.

### Long-term

- **Composability** — belief states can be shared between agents, merged, or persisted.
- **Auditability** — every belief has a source and timestamp. You can trace why an agent acted.
- **Alignment** — explicit beliefs are the foundation for corrigible, steerable agents.

---

## Why Not LangChain/LangGraph?

`belief-agent` is deliberately **minimal** and **framework-agnostic**:

- **Zero LangChain dependency** — works with raw LLM APIs or via LiteLLM.
- **Easy to compose** — use it inside LangGraph nodes, CrewAI agents, or standalone.
- **Educational** — the code is small (<400 lines of core logic), well-documented, and easy to fork.
- **Production-ready** — type hints, Pydantic v2, JSON serialization, 78% test coverage.

---

## Project Structure

```
belief-agent/
├── belief_agent/          # Core library
│   ├── __init__.py
│   ├── belief_state.py    # Belief + BeliefState
│   ├── agent.py           # BeliefAgent wrapper
│   ├── reflection.py      # Reflection logic
│   ├── negotiation.py     # Multi-objective negotiation
│   └── utils.py           # Prompt formatting helpers
├── examples/              # Runnable example scripts
│   ├── basic_agent.py
│   ├── long_horizon_planning.py
│   └── multi_objective_negotiation.py
├── notebooks/             # Markdown notebooks
│   ├── 01_quickstart.md
│   ├── 02_long_horizon.md
│   └── 03_negotiation.md
├── tests/                 # Pytest suite
│   ├── test_belief_state.py
│   ├── test_agent.py
│   └── test_reflection.py
├── README.md
├── pyproject.toml
├── LICENSE
├── .gitignore
└── requirements-dev.txt
```

---

## Development

```bash
git clone https://github.com/nulllabtests/belief-agent
cd belief-agent
pip install -e ".[litellm]"
pip install -r requirements-dev.txt
pytest tests/ --cov=belief_agent
```

---

## Contributing

Contributions are welcome! Areas we'd love help with:

- More sophisticated contradiction detection (LLM-based judge)
- Additional LLM client adapters
- Persistence backends (file, database)
- Plugin system for custom reflection strategies

Please open an issue first to discuss your idea.

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
If you find this useful, <strong>star the repo</strong> ⭐ and share it with other LLM engineers.
</p>
