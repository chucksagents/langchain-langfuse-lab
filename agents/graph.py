import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Must run before importing langfuse — get_client()/CallbackHandler() read
# credentials from os.environ when constructed, not lazily.
from dotenv import load_dotenv
load_dotenv()

from tools.dispute_tools import get_transaction, get_customer, get_customer_transactions
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict, Annotated
import operator
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse = get_client()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# Model must be defined before it is used
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0
)

tools = [get_transaction, get_customer, get_customer_transactions]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant designed to assist customers in placing disputes."
                    )
                ]
                + state["messages"],
                config={"callbacks": [langfuse_handler]}
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"], config={"callbacks": [langfuse_handler]})
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent with SQLite-backed memory
_DB_PATH = Path(__file__).parent.parent / "data" / "stateful-memory" / "checkpoints.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
memory = SqliteSaver(_conn)
agent = agent_builder.compile(checkpointer=memory)


