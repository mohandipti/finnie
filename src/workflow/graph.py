"""Assembles Finnie's LangGraph workflow.

    START → load_context → input_guard → router ─┬─► <agent> ─┐
                                 │                ├─► <agent> ─┴─► synthesizer ─┐
                                 └────────────────┴─► refusal ──────────────────┴─► compliance_guard → update_memory → END

Input guardrails: personal data is removed from the message before it enters the graph, so it is
never stored; input_guard refuses over-long messages and prompt injection; the router refuses
off-topic and disallowed requests. A refused message gets a fixed reply from the refusal node.

Only agents passed in (or created by default) are registered, and the router is
restricted to them. Conversation state is kept per session by a checkpointer,
keyed by session id: MongoDB by default (`default_checkpointer`), so conversations
survive restarts and can be shared by several app instances.

An agent can pause a turn to ask the user something (LangGraph `interrupt()`): the
Goal Planning Agent asks how much of the portfolio counts toward a goal. The paused
turn is kept in the checkpointer and finished with `resume_stream`.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pymongo import MongoClient
from pymongo.errors import PyMongoError

if TYPE_CHECKING:
    from src.core.config import Settings

from src.agents.base_agent import BaseAgent
from src.core.input_guard import redact_personal_data
from src.core.models import AgentResponse, ExtractedEntities, Portfolio, Source, UserProfile
from src.workflow.nodes import (
    after_input_guard,
    compliance_guard,
    input_guard,
    load_context,
    make_synthesizer_node,
    refusal,
    update_memory,
)
from src.workflow.router import make_router_node, select_agents
from src.workflow.state import FinnieState

logger = logging.getLogger(__name__)


@dataclass
class FinnieReply:
    """What the UI needs from one turn."""

    content: str
    sources: list[Source] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    route_reason: str = ""
    agent_outputs: list[AgentResponse] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class StreamEvent:
    """One event from `FinnieGraph.ask_stream`."""

    type: Literal["status", "token", "interrupt", "final"]
    text: str = ""
    node: str | None = None
    reply: FinnieReply | None = None  # set on the "final" event
    data: dict | None = None  # set on the "interrupt" event: the question for the user


def state_serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that allows Finnie's Pydantic state types to be restored."""
    allowed = [("src.core.models", cls.__name__) for cls in (AgentResponse, ExtractedEntities, Portfolio, Source, UserProfile)]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def default_checkpointer(settings: "Settings | None" = None) -> BaseCheckpointSaver:
    """Where conversations are stored: MongoDB (expiring after `conversation_ttl_days`), else process memory.

    Falls back to memory if MongoDB can't be reached, so chat keeps working (conversations are
    then lost on restart). Collections: conversation_checkpoints and conversation_writes.
    """
    from src.core.config import get_settings

    settings = settings or get_settings()
    if settings.conversation_store == "memory":
        return InMemorySaver(serde=state_serializer())
    try:
        client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        saver = MongoDBSaver(
            client,
            db_name=settings.mongodb_db,
            checkpoint_collection_name="conversation_checkpoints",
            writes_collection_name="conversation_writes",
            ttl=settings.conversation_ttl_days * 24 * 3600,
            serde=state_serializer(),
        )
    except PyMongoError as exc:
        # Log the error type only: the message can include the connection string.
        logger.warning("conversations kept in memory: MongoDB unavailable (%s)", type(exc).__name__)
        return InMemorySaver(serde=state_serializer())
    logger.info("conversations stored in MongoDB, kept for %d days", settings.conversation_ttl_days)
    return saver


def default_agents() -> list[BaseAgent]:
    """The agents implemented so far, using the configured LLMs and the FAISS knowledge base."""
    from src.agents.finance_qa_agent import FinanceQAAgent
    from src.agents.goal_planning_agent import GoalPlanningAgent
    from src.agents.market_agent import MarketAgent
    from src.agents.news_agent import NewsAgent
    from src.agents.portfolio_agent import PortfolioAgent
    from src.agents.tax_agent import TaxAgent
    from src.data.market_data import MarketDataService
    from src.data.news_data import NewsService
    from src.rag.faiss_retriever import FaissRetriever

    retriever = FaissRetriever.from_settings()
    market_data = MarketDataService.from_settings()
    return [
        FinanceQAAgent(retriever=retriever),
        MarketAgent(retriever=retriever, market_data=market_data),
        PortfolioAgent(retriever=retriever, market_data=market_data),
        NewsAgent(news=NewsService.from_settings()),
        TaxAgent(retriever=retriever),
        GoalPlanningAgent(retriever=retriever, market_data=market_data),
    ]


class FinnieGraph:
    def __init__(
        self,
        agents: list[BaseAgent] | None = None,
        router_llm: BaseChatModel | None = None,
        synthesizer_llm: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.agents = {agent.name: agent for agent in (agents if agents is not None else default_agents())}
        if "finance_qa" not in self.agents:
            raise ValueError("finance_qa must be registered: it is the router's default and fallback agent")

        graph = StateGraph(FinnieState)
        graph.add_node("load_context", load_context)
        graph.add_node("input_guard", input_guard)
        graph.add_node("refusal", refusal)
        graph.add_node("router", make_router_node(router_llm, available_agents=list(self.agents)))
        for name, agent in self.agents.items():
            graph.add_node(name, agent.as_node())
        graph.add_node("synthesizer", make_synthesizer_node(synthesizer_llm))
        graph.add_node("compliance_guard", compliance_guard)
        graph.add_node("update_memory", update_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "input_guard")
        graph.add_conditional_edges("input_guard", after_input_guard, ["router", "refusal"])
        # fan-out: the selected agents run in parallel, or the refusal node if the router refused the message
        graph.add_conditional_edges("router", select_agents, [*self.agents, "refusal"])
        graph.add_edge("refusal", "compliance_guard")
        for name in self.agents:
            graph.add_edge(name, "synthesizer")  # fan-in: runs once after all selected agents finish
        graph.add_edge("synthesizer", "compliance_guard")
        graph.add_edge("compliance_guard", "update_memory")
        graph.add_edge("update_memory", END)

        self.app = graph.compile(checkpointer=checkpointer or default_checkpointer())
        logger.info("graph compiled with agents=%s", list(self.agents))

    def ask(
        self,
        session_id: str,
        query: str,
        profile: UserProfile | None = None,
        portfolio: Portfolio | None = None,
    ) -> FinnieReply | None:
        """Run one conversational turn and return the complete reply. History is kept per `session_id`.

        The single entry point a frontend calls for each user message (or use `ask_stream` to show
        the answer as it's generated). Two things matter: create the graph once and reuse it, and
        give each user a stable session_id.
        """
        reply = None
        for event in self.ask_stream(session_id, query, profile, portfolio):
            if event.type == "final":
                reply = event.reply
        return reply  # None if the turn paused for a question; see `resume_stream`

    def ask_stream(
        self,
        session_id: str,
        query: str,
        profile: UserProfile | None = None,
        portfolio: Portfolio | None = None,
    ) -> Iterator["StreamEvent"]:
        """Run one turn, yielding events as it progresses.

        - `status`: a step finished, e.g. routing ("Answering with: finance_qa").
        - `token`: the next piece of answer text. Only the step that writes the user-facing answer
          streams: the single selected agent, or the synthesizer when several agents answered.
        - `interrupt`: the turn paused to ask the user something (`data` holds the question). Nothing
          more is yielded; call `resume_stream` with the answer to finish the turn.
        - `final`: the complete `FinnieReply`, always last unless the turn paused. Its `content` is
          authoritative and includes the disclaimer; render it in place of the streamed text, since a
          retried or failed LLM call can make the streamed text differ.

        `portfolio` is always written to state, so None clears a portfolio the user has deleted.
        """
        query, redacted = redact_personal_data(query)
        inputs: dict = {"session_id": session_id, "user_query": query, "portfolio": portfolio, "redacted": redacted}
        if profile is not None:
            inputs["user_profile"] = profile
        if self.pending_question(session_id) is not None:
            # A new message while a question is open counts as skipping it; that answer is discarded.
            for _ in self.resume_stream(session_id, {"amount": 0}):
                pass
        yield from self._stream(inputs, session_id)

    def resume_stream(self, session_id: str, answer: dict) -> Iterator["StreamEvent"]:
        """Finish a paused turn with the user's answer, yielding the same events as `ask_stream`."""
        yield from self._stream(Command(resume=answer), session_id)

    def pending_question(self, session_id: str) -> dict | None:
        """The question a paused turn is waiting on, or None."""
        snapshot = self.app.get_state(self._config(session_id))
        return snapshot.interrupts[0].value if snapshot and snapshot.interrupts else None

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def _stream(self, graph_input: dict | Command, session_id: str) -> Iterator["StreamEvent"]:
        config = self._config(session_id)
        answering_node: str | None = None
        if isinstance(graph_input, Command):  # resuming: routing already happened
            answering_node = self._answering_node(self.app.get_state(config).values.get("route") or [])
        streamed: list[str] = []
        for mode, payload in self.app.stream(graph_input, config=config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                node, update = next(iter(payload.items()))
                if node == "router" and update and not update.get("blocked_reason"):
                    route = update.get("route") or ["finance_qa"]
                    answering_node = self._answering_node(route)
                    yield StreamEvent("status", text=f"Answering with: {', '.join(route)}", node=node)
            elif mode == "messages":
                chunk, metadata = payload
                if answering_node and metadata.get("langgraph_node") == answering_node:
                    text = chunk.text
                    if text:
                        streamed.append(text)
                        yield StreamEvent("token", text=text, node=answering_node)

        question = self.pending_question(session_id)
        if question is not None:
            yield StreamEvent("interrupt", text=question.get("question", ""), data=question)
            return

        state = self.app.get_state(config).values
        reply = FinnieReply(
            content=state.get("final_response", ""),
            sources=state.get("sources", []),
            agents=state.get("route", []),
            route_reason=state.get("route_reason", ""),
            agent_outputs=state.get("agent_outputs", []),
            errors=state.get("errors", []),
        )
        # The compliance guard appends the disclaimer after the answer is generated; stream that tail too.
        streamed_text = "".join(streamed)
        if streamed_text and reply.content.startswith(streamed_text) and len(reply.content) > len(streamed_text):
            yield StreamEvent("token", text=reply.content[len(streamed_text) :], node="compliance_guard")
        yield StreamEvent("final", reply=reply)

    @staticmethod
    def _answering_node(route: list[str]) -> str:
        return route[0] if len(route) == 1 else "synthesizer"

    def history(self, session_id: str) -> list:
        """The conversation messages stored for a session."""
        snapshot = self.app.get_state({"configurable": {"thread_id": session_id}})
        return snapshot.values.get("messages", []) if snapshot else []
