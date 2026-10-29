"""Assembles Finnie's LangGraph workflow.

    START → load_context → router ─┬─► <agent> ─┐
                                   └─► <agent> ─┴─► synthesizer → compliance_guard → update_memory → END

Only agents passed in (or created by default) are registered, and the router is
restricted to them. Conversation state is kept per session by a checkpointer,
keyed by session id.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.agents.base_agent import BaseAgent
from src.core.models import AgentResponse, ExtractedEntities, Portfolio, Source, UserProfile
from src.workflow.nodes import compliance_guard, load_context, make_synthesizer_node, update_memory
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

    type: Literal["status", "token", "final"]
    text: str = ""
    node: str | None = None
    reply: FinnieReply | None = None  # set on the "final" event


def state_serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that allows Finnie's Pydantic state types to be restored."""
    allowed = [("src.core.models", cls.__name__) for cls in (AgentResponse, ExtractedEntities, Portfolio, Source, UserProfile)]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def default_agents() -> list[BaseAgent]:
    """The agents implemented so far, using the configured LLMs and the FAISS knowledge base."""
    from src.agents.finance_qa_agent import FinanceQAAgent
    from src.rag.faiss_retriever import FaissRetriever

    return [FinanceQAAgent(retriever=FaissRetriever.from_settings())]


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
        graph.add_node("router", make_router_node(router_llm, available_agents=list(self.agents)))
        for name, agent in self.agents.items():
            graph.add_node(name, agent.as_node())
        graph.add_node("synthesizer", make_synthesizer_node(synthesizer_llm))
        graph.add_node("compliance_guard", compliance_guard)
        graph.add_node("update_memory", update_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "router")
        graph.add_conditional_edges("router", select_agents, list(self.agents))  # fan-out, run in parallel
        for name in self.agents:
            graph.add_edge(name, "synthesizer")  # fan-in: runs once after all selected agents finish
        graph.add_edge("synthesizer", "compliance_guard")
        graph.add_edge("compliance_guard", "update_memory")
        graph.add_edge("update_memory", END)

        self.app = graph.compile(checkpointer=checkpointer or InMemorySaver(serde=state_serializer()))
        logger.info("graph compiled with agents=%s", list(self.agents))

    def ask(
        self,
        session_id: str,
        query: str,
        profile: UserProfile | None = None,
        portfolio: Portfolio | None = None,
    ) -> FinnieReply:
        """Run one conversational turn and return the complete reply. History is kept per `session_id`.

        The single entry point a frontend calls for each user message (or use `ask_stream` to show
        the answer as it's generated). Two things matter: create the graph once and reuse it, and
        give each user a stable session_id.
        """
        reply = None
        for event in self.ask_stream(session_id, query, profile, portfolio):
            if event.type == "final":
                reply = event.reply
        return reply

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
        - `final`: the complete `FinnieReply`, always last. Its `content` is authoritative and includes
          the disclaimer; render it in place of the streamed text, since a retried or failed LLM call
          can make the streamed text differ.
        """
        inputs: dict = {"session_id": session_id, "user_query": query}
        if profile is not None:
            inputs["user_profile"] = profile
        if portfolio is not None:
            inputs["portfolio"] = portfolio
        config = {"configurable": {"thread_id": session_id}}

        answering_node: str | None = None
        streamed: list[str] = []
        for mode, payload in self.app.stream(inputs, config=config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                node, update = next(iter(payload.items()))
                if node == "router" and update:
                    route = update.get("route") or ["finance_qa"]
                    answering_node = route[0] if len(route) == 1 else "synthesizer"
                    yield StreamEvent("status", text=f"Answering with: {', '.join(route)}", node=node)
            elif mode == "messages":
                chunk, metadata = payload
                if answering_node and metadata.get("langgraph_node") == answering_node:
                    text = chunk.text
                    if text:
                        streamed.append(text)
                        yield StreamEvent("token", text=text, node=answering_node)

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

    def history(self, session_id: str) -> list:
        """The conversation messages stored for a session."""
        snapshot = self.app.get_state({"configurable": {"thread_id": session_id}})
        return snapshot.values.get("messages", []) if snapshot else []
