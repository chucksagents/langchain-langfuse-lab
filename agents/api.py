import json
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
    stream: bool | None = None


def run_agent(req: ChatCompletionRequest) -> str:
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
    return reply


def sse_chunks(reply: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    # LangGraph's agent.invoke() returns the full reply at once (no native
    # token streaming wired up here), so this emits it as a single content
    # chunk followed by a stop chunk — enough to satisfy a caller expecting
    # at least one generation chunk, without a false claim of token streaming.
    yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': reply}, None))}\n\n"
    yield f"data: {json.dumps(chunk({}, 'stop'))}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    reply = run_agent(req)

    if req.stream:
        return StreamingResponse(sse_chunks(reply, req.model), media_type="text/event-stream")

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
