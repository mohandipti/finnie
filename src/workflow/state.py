"""LangGraph workflow state.

Nodes receive the full state and return a partial dict of keys to update.
Keys annotated with a reducer are merged; all other keys are overwritten.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.core.models import AgentResponse, ExtractedEntities, Portfolio, Source, UserProfile


def append_or_reset(current: list[Any] | None, update: list[Any] | None) -> list[Any]:
    """Append updates (so parallel agents don't overwrite each other); an update of None clears the list.

    The checkpointer keeps state across turns, so per-turn lists are reset at the start of each turn.
    """
    if update is None:
        return []
    return [*(current or []), *update]


class FinnieState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]  # conversation history across turns
    user_query: str
    user_profile: UserProfile
    portfolio: Portfolio | None
    route: list[str]  # agents selected by router
    route_reason: str
    extracted_entities: ExtractedEntities  # tickers, amounts, horizon, holdings found by the router
    agent_outputs: Annotated[list[AgentResponse], append_or_reset]  # this turn's agent answers
    retrieved_docs: list[Source]
    final_response: str
    sources: list[Source]
    errors: Annotated[list[str], append_or_reset]  # this turn's errors
