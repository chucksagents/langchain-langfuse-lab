# Session Notes

## Overview

Setup and debugging of a LangGraph agent with Langfuse observability tracing.

---

## 1. Install `langfuse`

**Problem:** `from langfuse import get_client` raised `ModuleNotFoundError` because `langfuse` was not installed in the active Python environment (`/opt/homebrew/…/site-packages`).

**Fix:**
```bash
pip install langfuse
```

Installed version: `langfuse==4.14.4`

---

## 2. Fix broken `.invoke()` call in `llm_call`

**Problem:** [`agents/graph.py`](../agents/graph.py) line 52 had a syntax/logic bug — the messages list was wrapped in `{}`, forming an invalid set expression:

```python
# Before (broken)
model_with_tools.invoke(
    {[SystemMessage(...)] + state["messages"]},
    config={"callbacks": [langfuse_handler]}
)
```

**Fix:** Remove the `{}` wrapper so the first argument is a plain list, and move `config=` out as a clean keyword argument:

```python
# After (correct)
model_with_tools.invoke(
    [SystemMessage(...)] + state["messages"],
    config={"callbacks": [langfuse_handler]}
)
```

This matches the pattern shown in the [Langfuse LangChain integration docs](https://langfuse.com/integrations/frameworks/langchain).

---

## 3. Install `langgraph-checkpoint-sqlite`

**Problem:** `from langgraph.checkpoint.sqlite import SqliteSaver` raised `ModuleNotFoundError`. Since LangGraph v0.2, the SQLite checkpointer ships as a separate package (`langgraph-checkpoint-sqlite`) rather than being bundled with `langgraph` itself.

**Fix:**
```bash
pip install langgraph-checkpoint-sqlite
```

Installed version: `langgraph-checkpoint-sqlite==3.1.1`

---

## Current State

`agents/graph.py` is fully functional with:

- **LangGraph** `StateGraph` with `llm_call` → `tool_node` loop and `should_continue` router
- **SQLite-backed memory** via `SqliteSaver` persisted at `data/stateful-memory/checkpoints.db`
- **Langfuse tracing** via `CallbackHandler` passed into every `model.invoke()` call through `config={"callbacks": [langfuse_handler]}`
- **Tool binding** to `get_transaction` from `tools/dispute_tools.py`
- **Model:** `gpt-4o-mini` at `temperature=0`
