import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agents.graph import agent
from langfuse import get_client, propagate_attributes

langfuse = get_client()


def main():
    print("Dispute assistant ready. Type 'quit' or 'exit' to end the session.\n")

    # One session ID per CLI session; use a fixed user ID for the CLI operator
    session_id = str(uuid.uuid4())
    user_id = "cli-user"

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye.")
                break

            # propagate_attributes applies session_id/user_id to every
            # observation created in this scope (root span, generations,
            # tool calls) — in Langfuse v4's OTel-based model these live on
            # each observation, not just the trace, so span.update(...)
            # alone does not set them.
            with propagate_attributes(session_id=session_id, user_id=user_id):
                # Root span for this turn — the LLM/tool generations from
                # the graph nest under it as siblings, per Langfuse's
                # tracing best practices.
                with langfuse.start_as_current_observation(
                    name="handle-customer-message",
                    as_type="span",
                    input=user_input,
                ) as span:
                    # thread_id ties this invoke to the persistent SqliteSaver
                    # checkpoint for this conversation — reusing session_id
                    # means one CLI session is both one Langfuse session and
                    # one checkpointed thread. Only the new message is passed
                    # in; the checkpointer supplies prior turns for this
                    # thread_id automatically. Passing the full accumulated
                    # history here too would double it up with what the
                    # checkpointer already restores.
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config={"configurable": {"thread_id": session_id}}
                    )
                    last = result["messages"][-1]
                    span.update(output=last.content)

            print(f"\nAssistant: {last.content}\n")
    finally:
        # Flush all pending traces before the process exits
        langfuse.flush()


if __name__ == "__main__":
    main()
