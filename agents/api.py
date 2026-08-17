import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agents.graph import agent, MessagesState
from langfuse import get_client, propagate_attributes

langfuse = get_client()

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "dispute-agent"
    messages: list[ChatMessage]
    user: str | None = None
    thread_id: str | None = None


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    user_message = req.messages[-1].content

    # Orchestrate's external_chat provider doesn't manage thread continuity
    # itself; a thread_id/user field in the request lets a caller opt into
    # a persistent conversation, otherwise each call gets its own thread.
    thread_id = req.thread_id or req.user or str(uuid.uuid4())
    user_id = req.user or "orchestrate-external-agent"

    with propagate_attributes(session_id=thread_id, user_id=user_id):
        with langfuse.start_as_current_observation(
            name="handle-customer-message",
            as_type="span",
            input=user_message,
        ) as span:
            result = agent.invoke(
                MessagesState(messages=[HumanMessage(content=user_message)], llm_calls=0),
                config={"configurable": {"thread_id": thread_id}}
            )
            reply = result["messages"][-1].content
            span.update(output=reply)

    langfuse.flush()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }
