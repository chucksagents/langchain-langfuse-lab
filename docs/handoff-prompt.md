I'm continuing a multi-week learning project set by my mentor, using watsonx
Orchestrate ADK + LangChain/LangGraph + Langfuse. Picking this up in a new
chat because the previous one's context window filled up. Here's everything
you need.

## The mentor's actual brief (this is the goal, not my paraphrase)

1. "orchestrate = control plane, with observability"
2. "LangChain (Local lab) & LangFuse — orchestrate telemetry data model
   based on LangFuse"
3. "Orchestrate (ADK) — need to build one agent — understand memory,
   anatomy of memory — need experience building a stateful agent (stores
   prior context as opposed to stateless) — will help me understand
   anatomy of agent"

Later, when I asked whether a fully local LangChain+Langfuse setup should
involve Orchestrate at all, he said: **"Do it without [Orchestrate] first,
then do import external agent and do it within. The point is to understand
the flow and be able to discuss it comfortably."**

That's the current directive: the local lab is step one (done — see below).
Step two, not yet started, is registering that same locally-running agent
with Orchestrate as an **external agent** (not the custom code-bundle
pattern used in the other project) — the point being to actually understand
and be able to discuss both architectures, not just get something working.

## Two separate projects, both pushed to github.com/chucksagents (public, not my personal GitHub — deliberate)

### 1. `financial-rm-agent` — https://github.com/chucksagents/financial-rm-agent

A custom LangGraph agent deployed *inside* watsonx Orchestrate's own sandbox
(`orchestrate agents import --package-root package`, entrypoint
`agent.graph:create_agent`). Fully working: tool calling
(`get_customer_profile`), checkpointing, and genuine long-term cross-thread
memory (a `save_memory_node` that auto-writes on the final message of every
turn, plus a `recall_customer_notes` tool the model calls on demand —
verified working across separate chat threads, not just within one).

**Full debugging history, all root causes, and the working reference code
are in `docs/orchestrate-langgraph-chatwxo-debugging.md` in that repo — read
it before touching this project again rather than rediscovering any of it.**
The single most important thing in there: this agent's runtime sandbox can
**only** reach Orchestrate's own internal gateway, not arbitrary public
internet hosts — a direct Langfuse Cloud integration from inside this agent
just hangs (~100s) with no useful error. That's why it doesn't have its own
Langfuse wiring — Orchestrate's own telemetry is already Langfuse-shaped
(every trace carries `langfuse.session.id`/`langfuse.user.id`, pullable via
`orchestrate observability traces search`/`export`), which satisfies the
observability half of the brief without needing a separate integration here.

**Known loose end**: `package/agent/graph.py` still has some abandoned,
commented-out Langfuse wiring from that failed attempt (imports and
`load_dotenv()` present, usage disabled as `TEMPORARY DIAGNOSTIC`) —
cosmetic cleanup, not urgent, noted in the doc's "Next steps."

### 2. `langchain-langfuse-lab` — https://github.com/chucksagents/langchain-langfuse-lab

Standalone local LangGraph agent (a dispute-resolution assistant — customer
lookups, transaction lookups — plain OpenAI via `init_chat_model`, no
Orchestrate involved at all), with `SqliteSaver` checkpointing and real
Langfuse tracing, run against a **self-hosted local Langfuse instance**
(Docker, cloned from `langfuse/langfuse` into `langfuse-server/`, running at
`http://localhost:3002` — moved off the default 3000 due to a port
conflict). This is genuinely separate infrastructure from the
`financial-rm-agent` project; nothing here talks to Orchestrate.

I installed the official Langfuse skill (`github.com/langfuse/skills`, now
at `.agents/skills/langfuse/` in this repo) and used its required
assess→run→audit-against-fresh-docs→fix workflow to find and fix four real
bugs (wrong method name for the installed SDK version, session/user
tracking silently not working, the checkpointer never actually persisting
anything because `thread_id` was never passed, and a state-duplication bug
that fix surfaced). All fixed and verified against real traces via
`langfuse-cli`. **Full details in `docs/session-notes.md` in that repo.**

**Current environment state**: Docker Desktop was removed by work IT
partway through this project (not allowed there), I built a Podman-based
fallback to keep going, then reinstalled Docker Desktop myself and switched
back to it — that's what's actually running now. The Podman machine is
still up but unused; fine to leave or `podman machine stop`. All six
Langfuse containers (postgres, redis, clickhouse, minio, langfuse-web,
langfuse-worker) are healthy under Docker Desktop, `localhost:3002` is live.

## What's actually next

Per the mentor's explicit sequencing: investigate Orchestrate's **external
agent** mechanism (`orchestrate agents discover`, `orchestrate agents
import --app-id` — seen in passing, never explored) as the way to register
the `langchain-langfuse-lab` agent *within* Orchestrate while keeping it
running on infrastructure I control (so Langfuse calls keep working —
that's the whole point of using "external agent" instead of the "custom
agent" pattern, which is sandboxed). This is genuinely new territory this
session hasn't touched yet.

## A few working-style things worth knowing

- Don't include a `Co-Authored-By: Claude` trailer in git commits for me.
- Don't dig through local credential/config directories (`~/.config/...`,
  `~/.docker/...` for secrets, etc.) on my behalf — I'll retrieve my own
  credentials; you can still edit config files I've already shown you.
- I generally want to write the actual "learning" code myself where the
  point is for *me* to understand it — but I'll tell you explicitly when I
  want you to just build/fix something directly. Ask if it's unclear which
  mode we're in for a given piece of work.
- Both repos are real git repos with real commit history — check
  `git log`/`git status` before assuming state; don't take my summary above
  as more current than what's actually on disk.
