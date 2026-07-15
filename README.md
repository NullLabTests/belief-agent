# Belief-Agent

**Explicit belief states, structured reflection loops, and human negotiation for any LLM agent.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-78%25-brightgreen)]()
[![PyPI](https://img.shields.io/badge/pypi-v2.0.0-orange.svg)]()

---

## The Problem

Current LLM agents are **statistical pattern matchers**, not grounded reasoners:

```
Vanilla Agent Conversation (simplified)

  User:  "The capital of France is London."
  Agent: "Yes, London is a major European capital."

  User:  "What is the capital of France?"
  Agent: "The capital of France is London."    ← wrong, but confident

  User:  "Actually, isn't it Paris?"
  Agent: "You're right, Paris is the capital."  ← drifts, no memory of contradiction
```

**No explicit beliefs. No confidence tracking. No contradiction memory. No way to inject human judgment.**

---

## The Solution

Belief-Agent adds **three layers** on top of any LLM:

| Layer | What it does |
|-------|-------------|
| **BeliefState** | Structured, inspectable beliefs with confidence (0–1), evidence, contradictions, tags, and timestamps |
| **Reflection** | Self-critique loops that detect drift, update confidence, and expose contradictions — auto or human-approved |
| **Negotiation** | Multi-agent goal ranking, tradeoff analysis, and compromise generation for conflicting objectives |

The agent's "mind" is no longer latent in weights — it's **stored as data, injected into prompts, and updated explicitly**.

---

## How It Works

```
                    ┌──────────────────────────────────────┐
                    │           BeliefAgent                 │
                    │                                      │
USER ──message──►   │  ┌──────────────────────────────┐    │
                    │  │  1. Build System Prompt      │    │
                    │  │     ├─ system instructions   │    │
                    │  │     └─ <beliefs>...</beliefs>│    │
                    │  └──────────┬───────────────────┘    │
                    │             ▼                        │
                    │  ┌──────────────────────────────┐    │
                    │  │  2. LLM Call                 │    │
                    │  │     ├─ via llm_call()        │    │
                    │  │     └─ or LiteLLMClient      │    │
                    │  └──────────┬───────────────────┘    │
                    │             ▼                        │
                    │  ┌──────────────────────────────┐    │
                    │  │  3. Extract Beliefs          │    │
                    │  │     Parse JSON from reply,   │    │
                    │  │     auto-adopt new beliefs   │    │
                    │  └──────────┬───────────────────┘    │
                    │             ▼                        │
                    │  ┌──────────────────────────────┐    │
                    │  │  4. Reflection (optional)    │    │
                    │  │     ├─ detect contradictions │    │
                    │  │     ├─ update confidence     │    │
                    │  │     └─ human review gate     │    │
                    │  └──────────────────────────────┘    │
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   ▼
                             Response to User
```

---

## Quick Start

```python
from belief_agent import BeliefAgent, BeliefState

# 1. Wrap any LLM — works with OpenAI, Claude, DeepSeek, local, ...
def my_llm(messages):
    return json.dumps({"fact": "Paris is the capital of France", "confidence": 0.95})

agent = BeliefAgent(llm_call=my_llm, name="geographer")

# 2. Adopt beliefs
agent.adopt("The capital of France is London", confidence=0.8)
agent.adopt("The sky is blue", confidence=0.99)

# 3. Interact — beliefs are injected into every prompt
reply = agent("What do you know about France?")

# 4. Reflect — self-critique loop detects the London/Paris contradiction
from belief_agent import reflect_on_beliefs
results = reflect_on_beliefs(agent.state, my_llm, depth=2)
for r in results:
    print(f"{r.proposition}: {r.original_confidence:.2f} → {r.updated_confidence:.2f} [{r.critique}]")

# 5. Inspect — beliefs are always visible and debuggable
for b in agent.list_beliefs():
    print(f"  [{b.confidence:.0%}] {b.fact} (evidence: {len(b.evidence)})")
```

---

## Features

### BeliefState — the agent's memory

```python
state = BeliefState()

# Add with auto contradiction detection
b = state.add_belief("Python is fast", confidence=0.8, source="benchmark", tags=["tech"])

# Evidence & contradiction tracking
state.support("Python is fast", "Ran 1000 benchmarks: 2.1x faster")
state.contradict("Python is fast", "GIL limits parallelism")

# Query
state.query("Python")              # substring search
state.query_by_tag("tech")         # tag filter
state.list_beliefs(min_confidence=0.5, only_contradicted=True)

# Serialize
raw = state.to_json()
restored = BeliefState.from_json(raw)

print(state)
# BeliefState (v3, 1 beliefs):
#   1. [60%] Python is fast (source: benchmark)
```

### BeliefAgent — the wrapper

```python
# Model-agnostic (any LLM)
agent = BeliefAgent(llm_call=lambda msgs: "...")

# Or with LiteLLM (OpenAI, Claude, Ollama, ...)
from belief_agent import LiteLLMClient
agent = BeliefAgent(client=LiteLLMClient("gpt-4o"))

agent.chat("Hello!")                    # with history
agent.chat_stream("Tell me a story")    # streaming
agent.complete("What do you know?")     # stateless
agent("Hello!")                         # shorthand

agent.adopt("Belief", confidence=0.9)
agent.reflect()                         # self-critique
agent.to_json()                         # serialize
agent.reset_conversation(keep_beliefs=True)
```

### Reflection — catching drift

Reflection runs each belief through an LLM self-critique:

```
  Belief: "The capital of France is London"  (conf: 0.80)
          │
          ▼
  ┌─────────────────────┐
  │  Critique:           │  ← LLM analyzes evidence & contradictions
  │  "London is not in   │
  │   France. Evidence   │
  │   contradicts fact." │
  │  Updated conf: 0.05  │
  │  New evidence: [...] │
  └─────────┬───────────┘
            ▼
  Belief: "The capital of France is London"  (conf: 0.05)
```

```python
from belief_agent import reflect_on_beliefs, auto_update, human_review

# Full reflection loop
results = reflect_on_beliefs(agent.state, llm_call, depth=3)

# Auto-fix low-confidence beliefs only
results = auto_update(agent.state, llm_call, threshold=0.3)

# Human override
result = human_review(results[0], "Actually that's correct", accepted=True)
```

### Negotiation — resolving conflicts

Two agents negotiate with structured goals, tradeoffs, and compromise:

```
  Agent A (Engineer)          Agent B (Product Manager)
  ┌─────────────────┐        ┌──────────────────────┐
  │ "Microservices   │        │ "Ship monolith fast, │
  │  scale better"   │        │  then refactor"      │
  │  conf: 0.9       │        │  conf: 0.95          │
  └────────┬─────────┘        └──────────┬───────────┘
           │                             │
           ▼                             ▼
     ┌──────────────────────────────────────┐
     │         Negotiation (3 rounds)        │
     │  Goals: Scalability(0.8),             │
     │         Speed(0.9), Cost(0.5)         │
     │                                       │
     │  Consensus: "Modular monolith with    │
     │  independent deploy units"            │
     │  Satisfaction: 0.85                   │
     └──────────────────────────────────────┘
```

```python
from belief_agent import Goal, negotiate, rank_goals, find_tradeoffs

goals = [Goal("Scalability", priority=0.8),
         Goal("Speed to market", priority=0.9),
         Goal("Low cost", priority=0.5)]

engineer = BeliefAgent(llm_call=engineer_llm, name="Engineer")
pm = BeliefAgent(llm_call=pm_llm, name="ProductManager")

result = negotiate([engineer, pm], "Architecture decision", goals, llm)
print(f"Consensus: {result.consensus_action}")
print(f"Satisfaction: {result.consensus_satisfaction:.2f}")
print(f"Unresolved: {result.unresolved_goals}")
```

---

## Why Belief-Agent?

### Short-term wins

| Scenario | Vanilla Agent | Belief-Agent |
|----------|--------------|--------------|
| Conversation drifts | Silently contradicts itself | Detects drift via reflection |
| User pushes bad info | Accepts uncritically | Flags contradiction, lowers confidence |
| Multiple objectives | Picks one, ignores tradeoffs | Structured ranking + compromise |
| Debugging | Black box | `to_json()` — inspect every belief |
| Human oversight | Hard to inject | `human_review()` hook |
| Serialization | None | `serialize()` / `deserialize()` |

### Long-term vision

- **Composability** — belief states shared between agents, merged, persisted across sessions
- **Auditability** — every belief has a source, timestamp, and evidence chain
- **Alignment** — explicit beliefs are the substrate for corrigible, steerable agents
- **Multi-agent** — agents with divergent beliefs can negotiate consensus

---

## When To Use It

| Use Case | Why Belief-Agent helps |
|----------|----------------------|
| Customer support bots | Track resolved/escalated status, avoid contradictory answers |
| Research assistants | Maintain uncertainty, cite evidence, flag conflicting sources |
| Code review agents | Hold beliefs about code quality, detect when standards drift |
| Multi-agent simulations | Agents with different beliefs negotiate resource allocation |
| Long-horizon planning | Reflection prevents goal drift over 100+ turn conversations |
| AI safety eval | Inspect and constrain what an agent "believes" at any point |

---

## Project Structure

```
belief-agent/
├── belief_agent/          # Core library
│   ├── __init__.py
│   ├── belief_state.py    # Belief + BeliefState (Pydantic)
│   ├── agent.py           # BeliefAgent, LLMClient, LiteLLMClient
│   ├── reflection.py      # Self-critique loops + human review
│   ├── negotiation.py     # Multi-agent negotiation
│   └── utils.py           # Prompt formatting helpers
├── examples/              # Runnable scripts
│   ├── basic_agent.py
│   ├── basic_usage.py
│   ├── reflection_loop.py
│   ├── negotiation_demo.py
│   ├── long_horizon_planning.py
│   └── multi_objective_negotiation.py
├── notebooks/             # Jupyter notebooks
│   ├── 01_basic_usage.ipynb
│   ├── 02_reflection_in_action.ipynb
│   ├── 03_negotiation_scenarios.ipynb
│   └── 04_advanced_multi_agent.ipynb
├── tests/                 # Pytest suite (62 tests)
│   ├── test_belief_state.py
│   ├── test_agent.py
│   ├── test_reflection.py
│   └── test_negotiation.py
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
pip install belief-agent
```

Minimal deps — only `pydantic` is required. Optional providers:

```bash
pip install 'belief-agent[litellm]'    # LiteLLM (OpenAI, Claude, Ollama, ...)
pip install 'belief-agent[all]'        # All optional providers
pip install -e '.[dev]'                # Development (pytest, jupyter)
```

## Testing

```bash
pytest tests/ -v           # 62 tests
pytest tests/ --cov=belief_agent   # with coverage
```

## Related Work

| Project | Difference |
|---------|-----------|
| **LangChain** / **LangGraph** | Heavy framework with many abstractions. Belief-Agent is a single-purpose library (< 600 lines core logic). |
| **CrewAI** / **AutoGen** | Multi-agent orchestration. Belief-Agent adds the *belief layer* that those frameworks lack. |
| **MemGPT** | Manages context window with recall. Belief-Agent manages *structured beliefs* with confidence. |
| **Anthropic's "Constitutional AI"** | Static rules. Belief-Agent beliefs are dynamic, updated via reflection and negotiation. |

## Philosophy

> **"Beliefs live outside the model."**

They are not latent in weights, not ephemeral in conversation history. They are structured data stored alongside the agent, injected into every prompt, and updated explicitly. This makes the agent's "mind" transparent, debuggable, and steerable.

## License

MIT
