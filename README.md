# Belief-Agent

**Explicit belief states, structured reflection loops, and human negotiation for any LLM agent.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Belief-Agent is a lightweight Python framework that wraps any LLM (OpenAI, Anthropic, DeepSeek, local models, …) with a **belief state** that tracks confidence, evidence, and contradictions. It adds **structured reflection** (self-critique loops) and **multi-agent negotiation** (goal ranking, tradeoffs, compromises) — all with optional human-in-the-loop.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     BeliefAgent                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  BeliefState  (belief map with confidence,          │  │
│  │               evidence, contradictions, tags, IDs)  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  LLM Call    │  │  Reflection  │  │  Negotiation     │ │
│  │  (any model) │  │  (critique)  │  │  (ranking/trade) │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from belief_agent import BeliefAgent, BeliefState

# 1. Wrap any LLM
def my_llm(messages):
    # calls OpenAI / Anthropic / DeepSeek / etc.
    return "...response..."

agent = BeliefAgent(llm_call=my_llm, name="my-agent")

# 2. Adopt beliefs
agent.adopt("AGI is 10 years away", confidence=0.6)

# 3. Interact
reply = agent("What do you think about AI timelines?")

# 4. Reflect on beliefs
from belief_agent import reflect_on_beliefs
reflect_on_beliefs(agent.state, my_llm, depth=2)

# 5. Negotiate with another agent
from belief_agent import Goal, negotiate
other = BeliefAgent(llm_call=my_llm, name="other")
goals = [Goal("Accuracy"), Goal("Speed")]
result = negotiate([agent, other], "Model choice", goals, my_llm)
```

## Features

### BeliefState

| Method | Description |
|--------|-------------|
| `add_belief(fact, confidence, source, tags)` | Add a new belief with auto contradiction detection |
| `update_belief(id, fact, confidence, ...)` | Update by ID |
| `remove_belief(id)` | Remove by ID |
| `support(fact, evidence_item)` | Add evidence and raise confidence |
| `contradict(fact, statement)` | Add contradiction, halve confidence |
| `merge(other)` | Combine two belief states |
| `query(text)` / `query_by_tag(tag)` / `query_by_source(src)` | Search |
| `is_confident(fact)` / `is_contradicted(fact)` | Quick checks |
| `list_beliefs(min_conf, only_contradicted)` | Filtered listing |
| `to_dict()` / `from_dict()` | Serialize to/from dict |
| `serialize()` / `deserialize()` | JSON round-trip |

### BeliefAgent

| Method | Description |
|--------|-------------|
| `chat(message)` | Send message with history |
| `chat_stream(message)` | Streaming version |
| `complete(prompt)` | Stateless completion |
| `adopt(fact, confidence, ...)` | Add or merge a belief |
| `get_belief(fact)` | Look up by fact text |
| `update_belief(fact, confidence, ...)` | Update by fact |
| `remove_belief(fact)` | Remove by fact |
| `list_beliefs(min_conf, only_contradicted)` | Filtered listing |
| `reflect()` | Ask agent to self-critique |
| `reset_conversation(keep_beliefs)` | Clear history |

Also works with `LLMClient`/`LiteLLMClient` for provider-specific features and streaming.

### Reflection

| Function | Description |
|----------|-------------|
| `reflect(state, user_msg, response, client)` | Analyze conversation turn, propose belief updates |
| `reflect_on_beliefs(state, llm_call, depth)` | Structured self-critique loop |
| `auto_update(state, llm_call, threshold)` | Reflect on low-confidence only |
| `human_review(result, feedback, accepted)` | Override with human judgment |

### Negotiation

| Function | Description |
|----------|-------------|
| `rank_goals(goals, llm)` | Sort goals by priority |
| `negotiate(agents, issue, goals, llm)` | Multi-agent consensus |
| `find_tradeoffs(agent, goal_a, goal_b, llm)` | Identify tradeoffs |
| `suggest_compromise(proposals, llm)` | Generate compromise |
| `negotiate_goals(goals, client)` | V1-compatible conflict analysis |

## Comparison: Vanilla Agent vs Belief-Agent

| Aspect | Vanilla Agent | Belief-Agent |
|--------|--------------|--------------|
| State | Ephemeral chat history | Explicit belief map |
| Confidence | None | 0.0–1.0 per belief |
| Contradictions | Lost / overwritten | Tracked explicitly + auto-detected |
| Reflection | None (or manual) | Structured self-critique |
| Human feedback | Hard to inject | `human_review()` hook |
| Multi-agent | Ad-hoc prompting | `negotiate()` with goals, tradeoffs, consensus |
| Serialization | None | JSON round-trip |
| Model support | Provider-specific | Model-agnostic (`llm_call`) or LiteLLM |

## Examples

| Example | File | Description |
|---------|------|-------------|
| **Basic State** | [`examples/basic_agent.py`](examples/basic_agent.py) | Belief CRUD, contradictions, merge, serialization |
| **Basic Agent** | [`examples/basic_usage.py`](examples/basic_usage.py) | Agent with fake LLM, belief extraction |
| **Reflection** | [`examples/reflection_loop.py`](examples/reflection_loop.py) | Structured self-critique with human override |
| **Negotiation** | [`examples/negotiation_demo.py`](examples/negotiation_demo.py) | Two agents negotiate architecture choice |
| **Long-Horizon** | [`examples/long_horizon_planning.py`](examples/long_horizon_planning.py) | Vanilla vs belief-agent drift comparison |
| **Multi-Objective** | [`examples/multi_objective_negotiation.py`](examples/multi_objective_negotiation.py) | Conflicts, ranking, compromises |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_basic_usage.ipynb` | Core data model and agent wrapper |
| `notebooks/02_reflection_in_action.ipynb` | Reflection loops and auto-update |
| `notebooks/03_negotiation_scenarios.ipynb` | Goal ranking, tradeoffs, compromise |
| `notebooks/04_advanced_multi_agent.ipynb` | Combined multi-agent workflow |

## Installation

```bash
pip install belief-agent
```

For development:

```bash
git clone https://github.com/nulllabtests/belief-agent
cd belief-agent
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
