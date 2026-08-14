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

## 4. Install the official Langfuse skill and audit tracing against it

Installed via `npx skills add langfuse/skills --skill "langfuse"` — now at `.agents/skills/langfuse/`, symlinked for Claude Code at `.claude/skills/langfuse`. Used its required workflow (assess → verify baseline → run and self-audit a real trace against the fresh-fetched best-practices doc → fix → repeat) to find and fix the following, rather than relying on memory of the API.

---

## 5. `start_as_current_span` doesn't exist on the installed client

**Problem:** [`agents/chat.py`](../agents/chat.py) called `langfuse.start_as_current_span(...)`, which crashed immediately:

```
AttributeError: 'Langfuse' object has no attribute 'start_as_current_span'. Did you mean: 'score_current_span'?
```

**Fix:** The correct method on `langfuse==4.14.4` is `start_as_current_observation`, with an explicit `as_type="span"` for the root orchestration span (matches best practices — generations/tools should be typed distinctly from the orchestration step wrapping them):

```python
with langfuse.start_as_current_observation(
    name="handle-customer-message",
    as_type="span",
    input=user_input,
) as span:
    ...
```

---

## 6. `session_id`/`user_id` were silently not being set

**Problem:** `span.update(session_id=session_id, user_id=user_id)` ran without error, but every observation in the resulting trace showed `sessionId: ""` and `userId: ""` — confirmed via `langfuse-cli api observations list --json`.

**Root cause:** In Langfuse v4's OTel-based model, `update_current_trace(session_id=..., user_id=...)` (the v3 pattern) is gone. These attributes now need to be applied to *every* observation individually, which is what `propagate_attributes()` does as a context manager around the whole scope:

```python
from langfuse import propagate_attributes

with propagate_attributes(session_id=session_id, user_id=user_id):
    with langfuse.start_as_current_observation(...) as span:
        ...
```

**Verified fix:** re-ran and confirmed via the CLI that `sessionId`/`userId` are now populated on every observation in the trace (root span, both generations, and the tool call).

---

## 7. The checkpointer was never actually persisting anything

**Problem:** `agent.invoke({"messages": history})` in `chat.py` never passed a `config`, so LangGraph raised:

```
ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id
```

`agent` is compiled with a `SqliteSaver` checkpointer, but without a `thread_id` it has no key to save/load state under — meaning the "stateful memory" this project is meant to demonstrate wasn't actually being exercised at all before this fix.

**Fix:** reuse the per-CLI-session `session_id` as the LangGraph `thread_id` too — one session is both one Langfuse session and one checkpointed conversation:

```python
result = agent.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config={"configurable": {"thread_id": session_id}}
)
```

---

## 8. Fixing #7 surfaced a state-duplication bug

**Problem:** `chat.py` also manually accumulated the full conversation in a Python `history` list and passed the whole thing in on every turn. Once the checkpointer actually started restoring prior state per `thread_id` (from fix #7), this meant every prior turn's messages would appear twice — once from the checkpoint, once from the redundant manually-passed `history`.

**Fix:** removed the manual `history` accumulation entirely. Only the new `HumanMessage` for the current turn is passed in; the checkpointer supplies everything before it.

**Verified fix:** ran a two-turn conversation ("What is customer CUST-1003's email?" then "What did I just ask you?") — the agent correctly recalled the first question using only checkpointed state, no duplication.

---

## 9. Best-practices cleanup (from the fresh-fetched best-practices doc)

- Root trace span renamed to be verb-first: `dispute-agent-turn` → `handle-customer-message` (best-practices doc explicitly calls out verb-first naming, e.g. `classify-intent`, `generate-response`).
- Added `LANGFUSE_TRACING_ENVIRONMENT=development` to `.env`/`.env.example`, per the doc's explicit guidance to set `environment` to prevent test data mixing with anything else.
- `requirements.txt` was missing `langgraph` and `langgraph-checkpoint-sqlite` — both installed and imported, just never declared. Added.
- Reordered imports in `graph.py` so `load_dotenv()` runs before any `langfuse` import (the skill explicitly flags this as a common mistake — `get_client()`/`CallbackHandler()` read credentials from `os.environ` at construction time).
- `.gitignore` extended to exclude `data/stateful-memory/*.db` and `*.png` (runtime state and a generated graph image, not source).

---

## 10. Verification and one open item

Verified via `langfuse-cli api observations list` (with `--fields core,basic,model,usage,io` — `model`/`usage` are separate field groups, not included by default):

- **Nesting is correct**: root `SPAN` (`handle-customer-message`) with two `GENERATION`s and one `TOOL` call as direct children/siblings — tool calls sit alongside the generation that requested them, not nested under it, matching the best-practices doc exactly.
- **Token usage is tracked**: `usageDetails` populated correctly (e.g. `{input: 205, output: 18, total: 223}`).
- **Observation types are correct**: `GENERATION` for LLM calls, `TOOL` for tool calls, `SPAN` for the root.

**Open item, not resolved:** `providedModelName` and `totalCost` come back empty/`0` despite an `internalModelId` being matched. Confirmed the raw LangChain response does contain the correct model name (`response_metadata['model_name'] == 'gpt-4o-mini-2024-07-18'`), so this isn't a gap in what our code sends — looks like a self-hosted instance model-catalog matching nuance rather than a code bug. Not chased further since the core signal (tokens, nesting, session/user tracking) is all confirmed correct.

---

## Current State

`agents/graph.py` + `agents/chat.py` are fully functional with:

- **LangGraph** `StateGraph` with `llm_call` → `tool_node` loop and `should_continue` router
- **SQLite-backed memory** via `SqliteSaver`, correctly keyed by `thread_id` (= the CLI session ID) and persisted at `data/stateful-memory/checkpoints.db` — verified working across multiple turns with no state duplication
- **Langfuse tracing** via `CallbackHandler` (LLM/tool spans) plus a manual root `start_as_current_observation` span per turn (trace-level input/output), with `session_id`/`user_id` correctly applied to every observation via `propagate_attributes()`
- **Tool binding** to `get_transaction`, `get_customer`, `get_customer_transactions` from `tools/dispute_tools.py`
- **Model:** `gpt-4o-mini` at `temperature=0`, via `init_chat_model`
- **Langfuse skill** installed at `.agents/skills/langfuse/` (symlinked for Claude Code), used to audit and fix the above against fresh-fetched best practices
