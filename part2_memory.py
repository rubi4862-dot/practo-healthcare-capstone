"""
part2_memory.py — Part 2: LangChain session memory.

InMemoryChatMessageHistory stores each session's turns; RunnableWithMessageHistory
injects that history into our runnable on every call. The mock "LLM" uses the
history to resolve follow-ups like "what fee did I pay for it?" -> the APT id
mentioned earlier in the SAME session. A fresh session has no history, so the same
follow-up gets a request for the appointment ID instead.
(LangChain may print a deprecation warning for RunnableWithMessageHistory; that is expected.)

Run:  MOCK_LLM=1 python part2_memory.py
"""

import re

import config  # noqa: F401

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.build import build_pipeline

PIPELINE = build_pipeline()
SESSIONS: dict[str, InMemoryChatMessageHistory] = {}
RECORD_RE = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)
FOLLOW_UP_REFS = re.compile(r"\b(it|that|this|my appointment|the appointment)\b", re.IGNORECASE)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return SESSIONS.setdefault(session_id, InMemoryChatMessageHistory())


def mock_chat(inputs: dict) -> AIMessage:
    """inputs = {"input": <new question>, "history": [previous messages]}"""
    question, history = inputs["input"], inputs["history"]
    if not RECORD_RE.search(question) and FOLLOW_UP_REFS.search(question):
        earlier = [m.group(0).upper() for msg in history for m in RECORD_RE.finditer(msg.content)]
        if earlier:
            question = f"{question} (appointment {earlier[-1]})"     # resolved from memory
        elif re.search(r"\b(fee|status|pay|paid)\b", question, re.IGNORECASE):
            return AIMessage(content="Which appointment do you mean? Please share the ID (format APT-XXXX).")
    return AIMessage(content=PIPELINE.answer(question).answer)


chain = RunnableWithMessageHistory(
    RunnableLambda(mock_chat),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat(session_id: str, text: str) -> None:
    reply = chain.invoke({"input": text}, config={"configurable": {"session_id": session_id}})
    print(f"[{session_id}] USER: {text}\n[{session_id}] BOT : {reply.content}\n")


if __name__ == "__main__":
    print("=== Session A: multi-turn, memory carries the appointment ID ===\n")
    chat("session-A", "What is the status of APT-1024?")
    chat("session-A", "What fee did I pay for it?")
    chat("session-A", "Is a follow-up visit within 7 days free for that appointment?")
    print(f"session-A history now holds {len(get_session_history('session-A').messages)} messages.\n")

    print("=== Session B: fresh session, no memory of APT-1024 ===\n")
    chat("session-B", "What fee did I pay for it?")
    print(f"session-B history holds {len(get_session_history('session-B').messages)} messages.")
