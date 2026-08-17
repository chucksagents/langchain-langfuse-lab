"""
Generate an accurate, thorough agent_graph.png for the LangGraph dispute agent.

Layout (top-to-bottom):
  ┌─ Caller context band ─────────────────────────────────────────────────┐
  │  FastAPI /v1/chat/completions  │  CLI chat.py                          │
  └───────────────────────────────────────────────────────────────────────┘
           │ thread_id + HumanMessage
           ▼
  ┌─ Langfuse span: handle-customer-message ─────────────────────────────┐
  │                                                                       │
  │   ┌── MessagesState ──────────────────────────────────────────────┐  │
  │   │  messages: list[AnyMessage]   llm_calls: int                  │  │
  │   └───────────────────────────────────────────────────────────────┘  │
  │                                                                       │
  │   START ──► llm_call ──[tool_calls?]──► tool_node ──┐               │
  │                │                                     │               │
  │                │ (no tool calls)                     │               │
  │                ▼                                     │               │
  │              END ◄──────────────────────────────────┘               │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

  Tools panel (right):
    • get_customer(customer_id)
    • get_customer_transactions(customer_id)
    • get_transaction(transaction_id)

  Persistence panel (bottom):
    SqliteSaver  ←─  thread_id  ─►  checkpoints.db
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ──────────────────────────────────────────────────────────
C_BG         = "#ffffff"
C_SURFACE    = "#f7f8fa"
C_BORDER     = "#d0d7de"
C_NODE_FILL  = "#dde4f5"   # graph nodes
C_NODE_EDGE  = "#6b7ec4"
C_TOOL_FILL  = "#e6f4ea"   # tool boxes
C_TOOL_EDGE  = "#4caf6e"
C_STATE_FILL = "#fff8e1"   # state box
C_STATE_EDGE = "#f0b429"
C_SPAN_FILL  = "#fef3c7"   # langfuse span background
C_SPAN_EDGE  = "#f59e0b"
C_CALLER_FILL= "#ede9fe"
C_CALLER_EDGE= "#7c5cd8"
C_PERS_FILL  = "#fce7f3"
C_PERS_EDGE  = "#db2777"
C_ARROW      = "#374151"
C_COND_ARROW = "#6b7ec4"
C_TEXT       = "#1f2328"
C_MUTED      = "#57606a"
C_ACCENT     = "#3b82d4"

FIG_W, FIG_H = 14, 11

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


# ── helpers ─────────────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.25, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    return box

def label(ax, x, y, text, size=9, color=C_TEXT, weight="normal",
          ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily="monospace" if "`" in text else "sans-serif")

def mono(ax, x, y, text, size=8, color=C_MUTED, ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight="normal",
            ha=ha, va=va, zorder=zorder, fontfamily="monospace")

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5,
          arrowstyle="-|>", zorder=4, label_text=None, label_color=C_MUTED):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=arrowstyle, color=color,
                        lw=lw, connectionstyle="arc3,rad=0"),
        zorder=zorder
    )
    if label_text:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.07, my, label_text, fontsize=7.5, color=label_color,
                ha="left", va="center", zorder=zorder+1,
                style="italic")

def curved_arrow(ax, x0, y0, x1, y1, rad=0.35, color=C_ARROW, lw=1.5, zorder=4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=zorder
    )

def dashed_arrow(ax, x0, y0, x1, y1, color=C_COND_ARROW, lw=1.4, zorder=4, label_text=None):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=0"),
        zorder=zorder
    )
    if label_text:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx-0.12, my, label_text, fontsize=7.5, color=color,
                ha="right", va="center", zorder=zorder+1, style="italic")

def section_label(ax, x, y, text, color=C_MUTED, size=7.5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight="bold",
            ha="left", va="top", style="italic", zorder=6)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CALLER CONTEXT  (top band)
# ═══════════════════════════════════════════════════════════════════════════
CALLER_X, CALLER_Y, CALLER_W, CALLER_H = 0.5, 9.5, 13, 1.2
rounded_box(ax, CALLER_X, CALLER_Y, CALLER_W, CALLER_H,
            fc=C_CALLER_FILL, ec=C_CALLER_EDGE, lw=1.2, radius=0.3, zorder=1)
section_label(ax, CALLER_X+0.15, CALLER_Y+CALLER_H-0.08, "CALLERS")

# FastAPI box
API_BX, API_BY, API_BW, API_BH = 1.0, 9.65, 4.5, 0.8
rounded_box(ax, API_BX, API_BY, API_BW, API_BH,
            fc="#ede9fe", ec=C_CALLER_EDGE, lw=1.2, radius=0.2)
label(ax, API_BX+API_BW/2, API_BY+API_BH/2+0.1,
      "FastAPI  /v1/chat/completions", size=9, weight="bold")
mono(ax,  API_BX+API_BW/2, API_BY+API_BH/2-0.18,
      "agents/api.py  ·  stream or JSON response", size=7.5)

# CLI box
CLI_BX, CLI_BY, CLI_BW, CLI_BH = 6.5, 9.65, 3.5, 0.8
rounded_box(ax, CLI_BX, CLI_BY, CLI_BW, CLI_BH,
            fc="#ede9fe", ec=C_CALLER_EDGE, lw=1.2, radius=0.2)
label(ax, CLI_BX+CLI_BW/2, CLI_BY+CLI_BH/2+0.1,
      "CLI  chat.py", size=9, weight="bold")
mono(ax,  CLI_BX+CLI_BW/2, CLI_BY+CLI_BH/2-0.18,
      "agents/chat.py  ·  one thread_id per CLI session, reused across all turns", size=7.5)

# Orchestrate box
ORC_BX, ORC_BY, ORC_BW, ORC_BH = 10.5, 9.65, 2.7, 0.8
rounded_box(ax, ORC_BX, ORC_BY, ORC_BW, ORC_BH,
            fc="#ede9fe", ec=C_CALLER_EDGE, lw=1.2, radius=0.2)
label(ax, ORC_BX+ORC_BW/2, ORC_BY+ORC_BH/2+0.1,
      "IBM Orchestrate", size=9, weight="bold")
mono(ax,  ORC_BX+ORC_BW/2, ORC_BY+ORC_BH/2-0.18,
      "external_chat provider", size=7.5)


# ═══════════════════════════════════════════════════════════════════════════
# 2. LANGFUSE SPAN  (outer observation wrapper)
# ═══════════════════════════════════════════════════════════════════════════
SPAN_X, SPAN_Y, SPAN_W, SPAN_H = 0.5, 1.8, 9.2, 7.5
rounded_box(ax, SPAN_X, SPAN_Y, SPAN_W, SPAN_H,
            fc=C_SPAN_FILL, ec=C_SPAN_EDGE, lw=1.8, radius=0.4,
            alpha=0.5, zorder=1)
section_label(ax, SPAN_X+0.2, SPAN_Y+SPAN_H-0.08,
              "LANGFUSE SPAN  ·  handle-customer-message",
              color="#b45309", size=8)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MessagesState box
# ═══════════════════════════════════════════════════════════════════════════
STATE_X, STATE_Y, STATE_W, STATE_H = 1.1, 8.05, 7.8, 1.1
rounded_box(ax, STATE_X, STATE_Y, STATE_W, STATE_H,
            fc=C_STATE_FILL, ec=C_STATE_EDGE, lw=1.3, radius=0.2, zorder=2)
label(ax, STATE_X+STATE_W/2, STATE_Y+STATE_H/2+0.2,
      "MessagesState  (TypedDict)", size=9, weight="bold", color="#92400e")
mono(ax,  STATE_X+STATE_W/2, STATE_Y+STATE_H/2-0.1,
      "messages: Annotated[list[AnyMessage], operator.add]   ·   llm_calls: int",
      size=7.8, color="#78350f")


# ═══════════════════════════════════════════════════════════════════════════
# 4. GRAPH NODES  — START, llm_call, tool_node, END
# ═══════════════════════════════════════════════════════════════════════════
# START  (pill)
START_X, START_Y = 4.5, 7.3
start_box = mpatches.FancyBboxPatch(
    (START_X-0.9, START_Y-0.28), 1.8, 0.56,
    boxstyle="round,pad=0,rounding_size=0.28",
    linewidth=1.5, edgecolor="#374151", facecolor="#e5e7eb", zorder=3
)
ax.add_patch(start_box)
label(ax, START_X, START_Y, "__start__", size=9, weight="bold", color="#374151")

# llm_call node
LLM_X, LLM_Y, LLM_W, LLM_H = 2.5, 5.4, 4.2, 1.3
rounded_box(ax, LLM_X, LLM_Y, LLM_W, LLM_H,
            fc=C_NODE_FILL, ec=C_NODE_EDGE, lw=2.0, radius=0.25, zorder=3)
label(ax, LLM_X+LLM_W/2, LLM_Y+LLM_H/2+0.22,
      "llm_call", size=11, weight="bold")
mono(ax,  LLM_X+LLM_W/2, LLM_Y+LLM_H/2-0.05,
      "gpt-4o-mini  ·  temperature=0", size=8)
mono(ax,  LLM_X+LLM_W/2, LLM_Y+LLM_H/2-0.30,
      "model_with_tools.invoke()  +  Langfuse CallbackHandler", size=7.5)

# tool_node
TOOL_X, TOOL_Y, TOOL_W, TOOL_H = 2.5, 3.5, 4.2, 1.1
rounded_box(ax, TOOL_X, TOOL_Y, TOOL_W, TOOL_H,
            fc=C_NODE_FILL, ec=C_NODE_EDGE, lw=2.0, radius=0.25, zorder=3)
label(ax, TOOL_X+TOOL_W/2, TOOL_Y+TOOL_H/2+0.18,
      "tool_node", size=11, weight="bold")
mono(ax,  TOOL_X+TOOL_W/2, TOOL_Y+TOOL_H/2-0.10,
      "executes tool_calls from last AIMessage  ·  emits ToolMessages", size=7.8)

# END  (pill)
END_X, END_Y = 4.6, 2.55
end_box = mpatches.FancyBboxPatch(
    (END_X-0.9, END_Y-0.28), 1.8, 0.56,
    boxstyle="round,pad=0,rounding_size=0.28",
    linewidth=1.5, edgecolor="#374151", facecolor="#e5e7eb", zorder=3
)
ax.add_patch(end_box)
label(ax, END_X, END_Y, "__end__", size=9, weight="bold", color="#374151")


# ═══════════════════════════════════════════════════════════════════════════
# 5. EDGES
# ═══════════════════════════════════════════════════════════════════════════
# __start__ ──► llm_call
arrow(ax, START_X, START_Y-0.28, LLM_X+LLM_W/2, LLM_Y+LLM_H, color=C_ARROW, lw=1.8)

# llm_call ──► tool_node  (dashed, conditional)
dashed_arrow(ax, LLM_X+LLM_W/2, LLM_Y,
             TOOL_X+TOOL_W/2, TOOL_Y+TOOL_H,
             label_text="tool_calls present")

# llm_call ──► __end__  (dashed, conditional)
dashed_arrow(ax, LLM_X, LLM_Y+LLM_H/2,
             END_X-0.9, END_Y,
             label_text="no tool_calls")

# tool_node ──► llm_call  (curved back-edge, right side)
curved_arrow(ax,
             TOOL_X+TOOL_W, TOOL_Y+TOOL_H/2,
             LLM_X+LLM_W,   LLM_Y+LLM_H/2,
             rad=-0.45, color=C_ACCENT, lw=1.8)
ax.text(LLM_X+LLM_W+1.05, (LLM_Y+LLM_H/2 + TOOL_Y+TOOL_H/2)/2,
        "loop back", fontsize=7.5, color=C_ACCENT,
        ha="left", va="center", style="italic", zorder=5)

# IBM Orchestrate ──► FastAPI box  (HTTP, stays inside caller band)
ax.annotate(
    "", xy=(API_BX+API_BW, ORC_BY+ORC_BH/2),
    xytext=(ORC_BX, ORC_BY+ORC_BH/2),
    arrowprops=dict(arrowstyle="-|>", color=C_CALLER_EDGE, lw=1.4,
                    connectionstyle="arc3,rad=0"),
    zorder=4
)
ax.text((API_BX+API_BW + ORC_BX)/2, ORC_BY+ORC_BH/2+0.18,
        "HTTP POST", fontsize=7.5, color=C_CALLER_EDGE,
        ha="center", va="bottom", style="italic", zorder=5)

# FastAPI ──► START
arrow(ax,
      API_BX+API_BW/2, CALLER_Y,
      START_X-0.3, START_Y+0.28,
      color=C_CALLER_EDGE, lw=1.6,
      label_text=" HumanMessage + thread_id",
      label_color=C_CALLER_EDGE)

# CLI ──► START
arrow(ax,
      CLI_BX+CLI_BW/2, CALLER_Y,
      START_X+0.3, START_Y+0.28,
      color=C_CALLER_EDGE, lw=1.6)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TOOL DEFINITIONS panel  (right side)
# ═══════════════════════════════════════════════════════════════════════════
TP_X, TP_Y, TP_W, TP_H = 10.1, 3.0, 3.5, 5.5
rounded_box(ax, TP_X, TP_Y, TP_W, TP_H,
            fc=C_TOOL_FILL, ec=C_TOOL_EDGE, lw=1.5, radius=0.3, zorder=2)
section_label(ax, TP_X+0.2, TP_Y+TP_H-0.1,
              "TOOLS  ·  tools/dispute_tools.py",
              color="#166534", size=8)

tools_info = [
    ("get_customer", "customer_id: str → dict",
     "customer profile, name, account status"),
    ("get_customer_transactions", "customer_id: str → list",
     "all card transactions for a customer"),
    ("get_transaction", "transaction_id: str → dict",
     "single transaction detail, amount, merchant"),
]
for i, (name, sig, desc) in enumerate(tools_info):
    ty = TP_Y + TP_H - 1.3 - i*1.55
    inner_box = rounded_box(ax, TP_X+0.2, ty-0.65, TP_W-0.4, 1.3,
                             fc="#ffffff", ec=C_TOOL_EDGE, lw=1.0, radius=0.15, zorder=3)
    label(ax, TP_X+TP_W/2, ty+0.3,  f"@tool  {name}", size=8.5,
          weight="bold", color="#166534")
    mono(ax,  TP_X+TP_W/2, ty+0.05, sig, size=7.5, color="#374151")
    mono(ax,  TP_X+TP_W/2, ty-0.2,  desc, size=7.2, color=C_MUTED)

# dashed connector: tool_node → tools panel
ax.annotate(
    "", xy=(TP_X, TP_Y+TP_H/2),
    xytext=(TOOL_X+TOOL_W, TOOL_Y+TOOL_H/2),
    arrowprops=dict(arrowstyle="-|>", color=C_TOOL_EDGE, lw=1.4,
                    linestyle="dashed",
                    connectionstyle="arc3,rad=0"),
    zorder=4
)
ax.text((TOOL_X+TOOL_W + TP_X)/2, TOOL_Y+TOOL_H/2+0.18,
        "dispatches", fontsize=7.5, color=C_TOOL_EDGE,
        ha="center", va="bottom", style="italic", zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# 7. PERSISTENCE panel  (bottom)
# ═══════════════════════════════════════════════════════════════════════════
PERS_X, PERS_Y, PERS_W, PERS_H = 0.5, 0.3, 9.2, 1.35
rounded_box(ax, PERS_X, PERS_Y, PERS_W, PERS_H,
            fc=C_PERS_FILL, ec=C_PERS_EDGE, lw=1.5, radius=0.25, zorder=2)
section_label(ax, PERS_X+0.2, PERS_Y+PERS_H-0.08,
              "PERSISTENCE  ·  SqliteSaver  (langgraph-checkpoint-sqlite)", color="#9d174d", size=8)

label(ax, PERS_X+PERS_W/2, PERS_Y+0.5,
      "SqliteSaver  ←─  thread_id  ─→  data/stateful-memory/checkpoints.db",
      size=9, weight="bold", color="#831843")
mono(ax,  PERS_X+PERS_W/2, PERS_Y+0.22,
      "Multi-turn memory  ·  state reloaded per call  ·  shared across CLI & API server",
      size=7.8, color="#9d174d")

# __end__ ──► persistence
arrow(ax, END_X, END_Y-0.28,
      PERS_X+PERS_W/2-1.0, PERS_Y+PERS_H,
      color=C_PERS_EDGE, lw=1.4,
      label_text=" checkpoint saved", label_color=C_PERS_EDGE)


# ═══════════════════════════════════════════════════════════════════════════
# 8. LANGFUSE instrumentation callout  (right of span, outside)
# ═══════════════════════════════════════════════════════════════════════════
LF_X, LF_Y, LF_W, LF_H = 10.1, 0.3, 3.5, 2.5
rounded_box(ax, LF_X, LF_Y, LF_W, LF_H,
            fc="#fef9c3", ec="#f59e0b", lw=1.4, radius=0.25, zorder=2)
section_label(ax, LF_X+0.2, LF_Y+LF_H-0.1,
              "LANGFUSE TRACING", color="#92400e", size=8)

lf_items = [
    "propagate_attributes(session_id, user_id)",
    "start_as_current_observation()",
    "  └─ ChatOpenAI  (generation)",
    "       └─ tool call  (tool span)",
    "             └─ ChatOpenAI  (generation)",
    "CallbackHandler on every invoke()",
]
for i, txt in enumerate(lf_items):
    mono(ax, LF_X+0.3, LF_Y+LF_H-0.55-i*0.33,
         txt, size=7.5, color="#78350f", ha="left")

# connector from span label to Langfuse callout
ax.annotate(
    "", xy=(LF_X, LF_Y+LF_H*0.7),
    xytext=(SPAN_X+SPAN_W, SPAN_Y+SPAN_H*0.2),
    arrowprops=dict(arrowstyle="-", color="#f59e0b", lw=1.2,
                    linestyle="dotted",
                    connectionstyle="arc3,rad=0"),
    zorder=3
)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Title + footer
# ═══════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, FIG_H-0.28,
        "LangGraph Dispute Resolution Agent — Architecture",
        fontsize=13, fontweight="bold", color=C_TEXT,
        ha="center", va="top", zorder=7)

ax.text(FIG_W/2, 0.12,
        "StateGraph (ReAct loop)  ·  gpt-4o-mini  ·  SqliteSaver  ·  Langfuse self-hosted  ·  FastAPI OpenAI-compatible endpoint",
        fontsize=7.5, color=C_MUTED, ha="center", va="bottom", zorder=7)

plt.tight_layout(pad=0.2)
plt.savefig("agent_graph.png", dpi=180, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print("Saved agent_graph.png")
