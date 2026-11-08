"""
part4_autogen.py — Part 4: AutoGen RoundRobinGroupChat review step.

  Policy_Compliance_Reviewer -> checks every claim in a draft against the context
  Final_Editor               -> output_content_type=Verdict (approved, final_answer, reason)

Team: RoundRobinGroupChat(max_turns=2, custom_message_types=[StructuredMessage[Verdict]]).
(AutoGen agent names must be Python identifiers, so the hyphens in
"Policy-Compliance-Reviewer" / "Final-Editor" become underscores.)

Case 1: a grounded draft from the pipeline        -> approved unchanged
Case 2: the same draft + an injected ungrounded claim -> revised

MockChatClient is a deterministic autogen_core ChatCompletionClient: it reads the
conversation and calls core.mock_llm.review_draft / edit_verdict.

Run:  MOCK_LLM=1 python part4_autogen.py
"""

import asyncio
import re
from typing import AsyncGenerator, Mapping, Sequence

import config  # noqa: F401

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import (ChatCompletionClient, CreateResult, LLMMessage, ModelInfo,
                                 RequestUsage)

from core import mock_llm
from core.governance import estimate_tokens
from core.schemas import Verdict
from rag.build import build_pipeline

DRAFT_RE = re.compile(r"DRAFT:\s*<<<(.+?)>>>", re.DOTALL)
CONTEXT_RE = re.compile(r"CONTEXT:\s*<<<(.+?)>>>", re.DOTALL)


class MockChatClient(ChatCompletionClient):
    """Deterministic model client. `role` picks which mock prompt it plays."""

    def __init__(self, role: str):
        self._role = role
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages: Sequence[LLMMessage], *, tools=(), json_output=None,
                     extra_create_args: Mapping = {}, cancellation_token=None, **kwargs) -> CreateResult:
        text = "\n".join(m.content for m in messages if isinstance(getattr(m, "content", None), str))
        draft = DRAFT_RE.search(text).group(1).strip()
        context = CONTEXT_RE.search(text).group(1).strip()

        if self._role == "reviewer":
            content = mock_llm.review_draft(draft, context)
        else:  # editor: find the reviewer's message in the conversation
            review = next(m.content for m in reversed(messages)
                          if isinstance(getattr(m, "content", None), str) and m.content.startswith("REVIEW:"))
            content = mock_llm.edit_verdict(draft, review, context).model_dump_json()

        self._last = RequestUsage(prompt_tokens=estimate_tokens(text),
                                  completion_tokens=estimate_tokens(content))
        self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + self._last.prompt_tokens,
                                   completion_tokens=self._total.completion_tokens + self._last.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=self._last, cached=False)

    async def create_stream(self, messages, *, tools=(), json_output=None, extra_create_args={},
                            cancellation_token=None, **kwargs) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, json_output=json_output)

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages, *, tools=()) -> int:
        return sum(estimate_tokens(str(getattr(m, "content", ""))) for m in messages)

    def remaining_tokens(self, messages, *, tools=()) -> int:
        return 8192 - self.count_tokens(messages)

    @property
    def capabilities(self):          # older autogen versions read this
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         family="unknown", structured_output=True)


def build_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=MockChatClient("reviewer"),
        system_message="Check every sentence of the DRAFT against the CONTEXT. "
                       "List unsupported claims. You have no tools.",
    )
    editor = AssistantAgent(
        name="Final_Editor",
        model_client=MockChatClient("editor"),
        system_message="Using the reviewer's findings, approve the draft unchanged or remove "
                       "unsupported claims. Reply as a Verdict.",
        output_content_type=Verdict,
    )
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[Verdict]])


async def review(draft: str, context: str, label: str) -> Verdict:
    print("=" * 78 + f"\n{label}\n" + "=" * 78)
    team = build_team()
    task = f"Review this support answer.\nDRAFT: <<<{draft}>>>\nCONTEXT: <<<{context}>>>"
    result = await team.run(task=task)
    for msg in result.messages:
        body = msg.content.model_dump_json() if hasattr(msg.content, "model_dump_json") else msg.content
        if msg.source == "user":
            body = f"DRAFT: {draft}"
        print(f"[{msg.source}] ({type(msg).__name__}) {body}\n")
    print(f"stop_reason: {result.stop_reason}")
    verdict = result.messages[-1].content
    assert isinstance(verdict, Verdict)
    Verdict.model_validate(verdict.model_dump())          # schema check in code
    print(f"VERDICT -> approved={verdict.approved}\n  final_answer: {verdict.final_answer}\n"
          f"  reason: {verdict.reason}\n")
    return verdict


async def main():
    pipe = build_pipeline()
    q = "How much does a cardiology consultation cost?"
    resp = pipe.answer(q)
    context = pipe.last_context.strip()

    v1 = await review(resp.answer, context, "CASE 1: grounded draft -> expect approve unchanged")
    assert v1.approved and v1.final_answer == resp.answer

    injected = resp.answer + " Cardiology consultations are always free for senior citizens."
    v2 = await review(injected, context, "CASE 2: injected ungrounded claim -> expect revise")
    assert not v2.approved and "senior citizens" not in v2.final_answer
    print("Both cases behaved as expected.")


if __name__ == "__main__":
    asyncio.run(main())
