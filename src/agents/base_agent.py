"""Base class for Finnie's specialized agents.

`run()` is a template method: retrieve knowledge base context, pre-fetch the
data the query obviously needs, build the prompt, call the LLM and package an
AgentResponse. Subclasses override only the steps they need. Any exception is
caught and turned into an AgentResponse with `error` set, so one failing agent
never crashes the graph.

Tool use is hybrid. `prefetch_data()` runs in code before the LLM is called,
so common queries need a single LLM call. Agents that also return tools from
`get_tools()` let the LLM request more data in a bounded tool-calling loop for
open-ended questions (e.g. comparing a stock with its competitors).
"""

import json
import logging
import time
from abc import ABC
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.core.models import AgentResponse, KnowledgeLevel, RetrievedDocument, Source, UserProfile
from src.rag.retriever import Retriever
from src.workflow.state import FinnieState

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "Sorry, I couldn't put together an answer to that just now. Please try again in a moment, "
    "or rephrase your question."
)

LEVEL_INSTRUCTIONS: dict[KnowledgeLevel, str] = {
    "beginner": (
        "The user is a beginner. Use everyday language and short sentences, explain every term, "
        "prefer analogies and simple examples over formulas."
    ),
    "intermediate": (
        "The user has intermediate knowledge. You can use common financial terms with a brief "
        "reminder of their meaning, and include simple numbers or formulas where they help."
    ),
    "advanced": (
        "The user is advanced. Be precise and concise, use standard terminology, and include "
        "technical detail, formulas and caveats where relevant."
    ),
}

def _to_text(result: Any) -> str:
    """Serialize a tool result for a ToolMessage (Pydantic models, dicts, lists or plain values)."""
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result, default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o))


TOOL_USE_INSTRUCTION = (
    "Data for the question has already been fetched and is included in the user's message. "
    "Call a tool only if you need data that is missing (for example another ticker or time period); "
    "otherwise answer directly."
)


class BaseAgent(ABC):
    name: str
    description: str  # used by the router to decide when this agent applies
    system_prompt: str

    history_turns: int = 6  # recent conversation messages included in the prompt
    retrieval_k: int = 5
    min_score: float | None = None  # chunks below this relevance are discarded; None uses the configured default
    retrieval_categories: list[str] | None = None  # None searches the whole knowledge base
    max_tool_rounds: int = 3  # LLM calls allowed to request tools before a final answer is forced

    def __init__(self, llm: BaseChatModel | None = None, retriever: Retriever | None = None, max_attempts: int = 3):
        if llm is None:
            from src.core.config import get_llm

            llm = get_llm(self.name)  # this agent's own model, configurable via AGENT_LLM
        self.base_llm = llm
        self.max_attempts = max_attempts
        self.llm = llm.with_retry(stop_after_attempt=max_attempts)
        self.retriever = retriever
        if self.min_score is None:
            from src.core.config import min_relevance_score

            self.min_score = min_relevance_score()

    # ---- template method -------------------------------------------------

    def run(self, state: FinnieState) -> AgentResponse:
        started = time.perf_counter()
        try:
            docs = self.retrieve_context(state)
            data = self.prefetch_data(state)
            messages = self.build_messages(state, docs, data)
            content, tool_results = self.generate(messages)
            if tool_results:
                data = {**(data or {}), "tool_results": tool_results}
            response = self.build_response(state, content, docs, data)
        except Exception as exc:
            logger.exception("agent=%s session=%s failed", self.name, state.get("session_id"))
            response = AgentResponse(agent=self.name, content=FALLBACK_MESSAGE, error=str(exc) or type(exc).__name__)
        logger.info(
            "agent=%s session=%s latency_ms=%.0f error=%s",
            self.name, state.get("session_id"), (time.perf_counter() - started) * 1000, response.error,
        )
        return response

    def as_node(self):
        """LangGraph node: run the agent and append its response to `agent_outputs`."""

        def node(state: FinnieState) -> dict:
            response = self.run(state)
            update: dict[str, Any] = {"agent_outputs": [response]}
            if response.error:
                update["errors"] = [f"{self.name}: {response.error}"]
            return update

        node.__name__ = self.name
        return node

    # ---- overridable steps -----------------------------------------------

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        """Knowledge base chunks for this query. Agents that use RAG override or call `retrieve`."""
        return []

    def prefetch_data(self, state: FinnieState) -> dict[str, Any] | None:
        """Data the query obviously needs, fetched in code (quotes, metrics, projections).

        Returned data goes into the prompt and to the UI for charts.
        """
        return None

    def get_tools(self) -> list[BaseTool]:
        """Tools the LLM may call for data that wasn't pre-fetched. No tools means a single LLM call."""
        return []

    def build_messages(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> list[BaseMessage]:
        system = self.system_prompt + "\n\n" + self.adapt_to_level(state.get("user_profile"))
        if self.get_tools():
            system += "\n\n" + TOOL_USE_INSTRUCTION
        history = self.recent_history(state)
        return [SystemMessage(content=system), *history, HumanMessage(content=self.build_user_message(state, docs, data))]

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        return state.get("user_query", "")

    def generate(self, messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
        """Return the answer text and the results of any tools the LLM called."""
        tools = self.get_tools()
        if tools:
            reply, tool_results = self._run_tool_loop(messages, tools)
        else:
            reply, tool_results = self.llm.invoke(messages), []
        text = reply.text.strip()
        if not text:
            raise ValueError("LLM returned an empty response")
        return text, tool_results

    def _run_tool_loop(
        self, messages: list[BaseMessage], tools: list[BaseTool]
    ) -> tuple[AIMessage, list[dict[str, Any]]]:
        """Let the LLM request tools for up to `max_tool_rounds` calls, then force a final answer."""
        llm_with_tools = self.base_llm.bind_tools(tools).with_retry(stop_after_attempt=self.max_attempts)
        tools_by_name = {tool.name: tool for tool in tools}
        conversation = list(messages)
        results: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            reply = llm_with_tools.invoke(conversation)
            if not reply.tool_calls:
                return reply, results
            conversation.append(reply)
            for call in reply.tool_calls:
                result = self._execute_tool(tools_by_name, call)
                results.append({"tool": call["name"], "args": call["args"], "result": result})
                conversation.append(ToolMessage(content=_to_text(result), tool_call_id=call["id"], name=call["name"]))

        logger.warning("agent=%s hit max_tool_rounds=%d; forcing a final answer", self.name, self.max_tool_rounds)
        return self.llm.invoke(conversation), results

    def _execute_tool(self, tools_by_name: dict[str, BaseTool], call: dict[str, Any]) -> Any:
        """Run one requested tool; errors are returned to the LLM as text so it can adapt."""
        tool = tools_by_name.get(call["name"])
        if tool is None:
            return f"Error: unknown tool '{call['name']}'."
        try:
            return tool.invoke(call["args"])
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the LLM
            logger.warning("agent=%s tool=%s failed: %s", self.name, call["name"], exc)
            return f"Error: {call['name']} failed ({exc})."

    def build_response(
        self, state: FinnieState, content: str, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> AgentResponse:
        return AgentResponse(
            agent=self.name,
            content=content,
            sources=self.format_sources(docs),
            data=data,
            confidence=max((d.score for d in docs), default=None),
        )

    # ---- shared helpers --------------------------------------------------

    def retrieve(self, query: str, categories: list[str] | None = None) -> list[RetrievedDocument]:
        """Retrieve relevant chunks, dropping those below `min_score`. Returns [] when no retriever is configured."""
        if self.retriever is None or not query.strip():
            return []
        docs = self.retriever.retrieve(query, k=self.retrieval_k, categories=categories)
        return [d for d in docs if d.score >= self.min_score]

    @staticmethod
    def adapt_to_level(profile: UserProfile | None) -> str:
        level = profile.knowledge_level if profile else "beginner"
        return LEVEL_INSTRUCTIONS[level]

    def recent_history(self, state: FinnieState) -> list[BaseMessage]:
        """Prior conversation turns, excluding the current query if it was already added to `messages`."""
        messages: Sequence[BaseMessage] = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == state.get("user_query"):
            messages = messages[:-1]
        return list(messages[-self.history_turns:]) if self.history_turns else []

    @staticmethod
    def format_context(docs: list[RetrievedDocument]) -> str:
        """Excerpts the LLM can cite as [1], [2], ...

        Numbers are per article, not per chunk, so they match the order of `format_sources()`:
        two excerpts from the same article are both [1].
        """
        numbers: dict[tuple[str, str | None], int] = {}
        blocks = []
        for doc in docs:
            n = numbers.setdefault((doc.source.title, doc.source.url), len(numbers) + 1)
            header = f"[{n}] {doc.source.title}"
            if doc.source.category:
                header += f" (category: {doc.source.category})"
            blocks.append(f"{header}\n{doc.content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def format_sources(docs: list[RetrievedDocument]) -> list[Source]:
        """One Source per distinct article, in retrieval order."""
        seen: set[tuple[str, str | None]] = set()
        sources = []
        for doc in docs:
            key = (doc.source.title, doc.source.url)
            if key not in seen:
                seen.add(key)
                sources.append(doc.source)
        return sources
