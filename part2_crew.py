"""
part2_crew.py — Part 2: a 3-agent CrewAI crew run with .kickoff() under MOCK_LLM.

  Retrieval Agent  -- tool: rag_search               (searches the KB collection)
  Lookup Agent     -- tool: check_appointment_status (the ONLY agent with record access)
  Composer Agent   -- no tools; writes the final JSON answer from the two outputs

Tools are handed out from core.governance.AGENT_TOOL_ALLOWLIST (least autonomy).
The final output is validated in code with SupportResponse.model_validate_json().

MockCrewLLM subclasses crewai.llms.base_llm.BaseLLM and speaks CrewAI's ReAct
text format ("Action: / Action Input: / Final Answer:"). Two pitfalls handled:
  1. CrewAI's own system prompt contains the text "Observation: the result of the
     action". We therefore only look for "Observation:" in messages that come AFTER
     the initial prompt, i.e. text the loop produced, never in the instructions.
  2. Tool arguments are built from each tool's declared args_schema field names
     (record_id / query), never by substring-matching the tool name
     ("rag_lookup"-style names would otherwise be confused with the lookup tool).

Run:  MOCK_LLM=1 python part2_crew.py
"""

import json
import re
from typing import Any

import config  # noqa: F401  (telemetry off before crewai import)

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from core import mock_llm
from core.governance import AGENT_TOOL_ALLOWLIST
from core.guardrails import mask_pii
from core.pipeline import policy_query
from core.schemas import AppointmentStatus, SupportResponse
from core.status_tool import check_appointment_status, describe_status
from rag.build import build_pipeline

TOOL_CALLS: list[dict] = []          # evidence that both tools were really invoked
PIPELINE = None                      # set in main(); gives the RAG tool its store + threshold

QUESTION_RE = re.compile(r"Customer question: <<<(.+?)>>>", re.DOTALL)
RECORD_RE = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)
CHUNK_RE = re.compile(r"\[source=(?P<doc>[\w.-]+) score=(?P<score>[\d.]+)\] (?P<text>.+)")


# ----------------------------------------------------------------------------- tools
class RagSearchInput(BaseModel):
    query: str = Field(..., description="The customer's question, used to search the policy KB.")


class RagSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = "Search the Practo policy knowledge base and return the top matching chunks."
    args_schema: type[BaseModel] = RagSearchInput

    def _run(self, query: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"query": query}})
        chunks = PIPELINE.retriever.query(PIPELINE.strategy, policy_query(query), PIPELINE.top_k)
        if not chunks or chunks[0]["score"] < PIPELINE.threshold:
            top = chunks[0]["score"] if chunks else 0.0
            return f"NO_RELEVANT_CONTEXT (top1={top:.4f} < threshold {PIPELINE.threshold})"
        return "\n".join(f"[source={c['doc_id']} score={c['score']:.4f}] {c['text']}" for c in chunks)


class StatusInput(BaseModel):
    record_id: str = Field(..., description="Appointment ID in the format APT-XXXX.")


class CheckAppointmentStatusTool(BaseTool):
    name: str = "check_appointment_status"
    description: str = "Return status, fee and escalation score for one appointment ID."
    args_schema: type[BaseModel] = StatusInput

    def _run(self, record_id: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"record_id": record_id}})
        return json.dumps(check_appointment_status(record_id))


ALL_TOOLS = {"rag_search": RagSearchTool(), "check_appointment_status": CheckAppointmentStatusTool()}


# ----------------------------------------------------------------------------- mock LLM
def _content(m: Any) -> str:
    c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    return c if isinstance(c, str) else json.dumps(c)


def _role(m: Any) -> str:
    return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")


class MockCrewLLM(BaseLLM):
    """Deterministic stand-in for a chat model, speaking CrewAI's ReAct format."""

    def __init__(self):
        super().__init__(model="mock-llm", temperature=0.0)

    # CrewAI asks these before using an LLM
    def supports_function_calling(self) -> bool:
        return False                  # forces the plain-text ReAct path we parse below

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # The initial prompt = system message(s) + the first user message.
        # Everything after it was produced by the agent loop (pitfall 1).
        first_user = next(i for i, m in enumerate(messages) if _role(m) == "user")
        prompt_text = "\n".join(_content(m) for m in messages[: first_user + 1])
        loop_text = "\n".join(_content(m) for m in messages[first_user + 1:])

        role = getattr(from_agent, "role", None) or self._role_from_prompt(prompt_text)
        question = (QUESTION_RE.search(prompt_text) or [None, ""])[1].strip()
        agent_tools = list(getattr(from_agent, "tools", None) or [])
        if not agent_tools:   # older CrewAI may not pass from_agent: read tool names from the prompt
            agent_tools = [ALL_TOOLS[n] for n in re.findall(r"Tool Name: (\w+)", prompt_text)
                           if n in ALL_TOOLS]

        observation = None
        if "Observation:" in loop_text:
            observation = loop_text.split("Observation:")[-1].strip()

        # ---- step 1: call this agent's single tool, args from its schema -----------
        if agent_tools and observation is None:
            tool = agent_tools[0]
            args = self._args_from_schema(tool, question)
            if args is None:          # e.g. Lookup Agent but no APT id in the question
                return ("Thought: The question contains no appointment ID, so there is nothing to look up.\n"
                        "Final Answer: NO_RECORD_ID")
            return (f"Thought: I should use {tool.name} to get the facts.\n"
                    f"Action: {tool.name}\nAction Input: {json.dumps(args)}")

        # ---- step 2: tool agents return the observation verbatim -------------------
        if observation is not None:
            return f"Thought: I now know the final answer\nFinal Answer: {observation}"

        # ---- Composer (no tools): build SupportResponse JSON from task context ----
        return "Thought: I now know the final answer\nFinal Answer: " + self._compose(question, prompt_text)

    @staticmethod
    def _role_from_prompt(prompt_text: str) -> str:
        m = re.search(r"You are (.+?)\.", prompt_text)
        return m.group(1) if m else "unknown"

    @staticmethod
    def _args_from_schema(tool, question: str) -> dict | None:
        """Pitfall 2: decide arguments from the declared schema fields, not the tool name."""
        fields = list(tool.args_schema.model_fields)
        args = {}
        for f in fields:
            if "record" in f:
                m = RECORD_RE.search(question)
                if not m:
                    return None
                args[f] = m.group(0).upper()
            elif f in ("query", "question", "text"):
                args[f] = question
        return args

    @staticmethod
    def _compose(question: str, prompt_text: str) -> str:
        chunks = [{"doc_id": m["doc"], "text": m["text"], "score": float(m["score"])}
                  for m in CHUNK_RE.finditer(prompt_text)]
        status = None
        m = re.search(r'\{"found".*?\}', prompt_text)
        if m:
            status = json.loads(m.group(0))

        parts, sources = [], []
        if status:
            parts.append(describe_status(status))
        if chunks:
            ans, sources = mock_llm.compose_grounded_answer(question, chunks)
            parts.append(ans)
        if not parts:
            resp = SupportResponse(answer=mock_llm.IDK_ANSWER, grounded=True, refused=True,
                                   refusal_reason="out_of_scope")
        else:
            resp = SupportResponse(
                answer=" ".join(parts), sources=sources, grounded=True,
                top1_score=chunks[0]["score"] if chunks else None,
                appointment=AppointmentStatus.model_validate(status) if status else None)
        return resp.model_dump_json()


# ----------------------------------------------------------------------------- crew
def build_crew(llm: BaseLLM) -> Crew:
    def tools_for(role):                       # least autonomy: tools come from the allowlist
        return [ALL_TOOLS[name] for name in sorted(AGENT_TOOL_ALLOWLIST[role])]

    retrieval = Agent(role="Retrieval Agent", goal="Find the policy passages that answer the question.",
                      backstory="You search the Practo knowledge base and never invent policy.",
                      tools=tools_for("Retrieval Agent"), llm=llm, allow_delegation=False, max_iter=3)
    lookup = Agent(role="Lookup Agent", goal="Fetch the appointment record mentioned in the question.",
                   backstory="You are the only agent allowed to read appointment records.",
                   tools=tools_for("Lookup Agent"), llm=llm, allow_delegation=False, max_iter=3)
    composer = Agent(role="Composer Agent", goal="Write a grounded answer as SupportResponse JSON.",
                     backstory="You only use the retrieved passages and the record lookup.",
                     tools=tools_for("Composer Agent"), llm=llm, allow_delegation=False, max_iter=2)

    t_retrieve = Task(description="Customer question: <<<{question}>>>\nRetrieve the relevant policy chunks.",
                      expected_output="Policy chunks with [source=...] tags, or NO_RELEVANT_CONTEXT.",
                      agent=retrieval)
    t_lookup = Task(description="Customer question: <<<{question}>>>\nLook up the appointment if an ID is given.",
                    expected_output="The appointment status JSON, or NO_RECORD_ID.", agent=lookup)
    t_compose = Task(description="Customer question: <<<{question}>>>\nCompose the final answer.",
                     expected_output="A JSON object matching the SupportResponse schema.",
                     agent=composer, context=[t_retrieve, t_lookup])
    return Crew(agents=[retrieval, lookup, composer], tasks=[t_retrieve, t_lookup, t_compose],
                process=Process.sequential, verbose=True)


def main():
    global PIPELINE
    PIPELINE = build_pipeline()
    crew = build_crew(MockCrewLLM())
    print("Agent -> tools (from AGENT_TOOL_ALLOWLIST):")
    for a in crew.agents:
        print(f"  {a.role:<16} {[t.name for t in a.tools] or '(no tools)'}")

    questions = [
        "What is the status of APT-1024 and is a follow-up within 7 days free?",
        "How much does a dermatology consultation cost?",
    ]
    for q in questions:
        q_masked, _ = mask_pii(q)
        TOOL_CALLS.clear()
        print("\n" + "=" * 78 + f"\nKICKOFF: {q_masked}\n" + "=" * 78)
        result = crew.kickoff(inputs={"question": q_masked})
        validated = SupportResponse.model_validate_json(result.raw)       # schema check in code
        print("\nTools invoked during this kickoff:")
        for c in TOOL_CALLS:
            print(f"  - {c['tool']}({c['args']})")
        print("\nValidated SupportResponse:")
        print(validated.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
