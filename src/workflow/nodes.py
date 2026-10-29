"""Workflow nodes other than the router and agents.

    load_context → router → agent(s) → synthesizer → compliance_guard → update_memory
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.core.disclaimers import ADVICE_CAVEAT, DISCLAIMER, find_advice_phrases, has_disclaimer, strip_disclaimer
from src.core.models import AgentResponse, Source
from src.data.sample_data import load_sample_profile
from src.workflow.state import FinnieState

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = (
    "Sorry, I couldn't answer that right now. Please try again in a moment, or rephrase your question."
)


def load_context(state: FinnieState) -> dict[str, Any]:
    """Start a turn: record the user's message and clear the previous turn's per-turn values."""
    query = (state.get("user_query") or "").strip()
    update: dict[str, Any] = {
        "messages": [HumanMessage(content=query)] if query else [],
        "agent_outputs": None,  # reset (see append_or_reset)
        "errors": None,
        "route": [],
        "route_reason": "",
        "sources": [],
        "final_response": "",
    }
    if not state.get("user_profile"):
        update["user_profile"] = load_sample_profile(state.get("session_id", "anonymous"))
    return update


def _merge_sources(outputs: list[AgentResponse]) -> list[Source]:
    seen: set[tuple[str, str | None]] = set()
    merged = []
    for output in outputs:
        for source in output.sources:
            if (source.title, source.url) not in seen:
                seen.add((source.title, source.url))
                merged.append(source)
    return merged


def make_synthesizer_node(llm: BaseChatModel | None = None, max_attempts: int = 3):
    """Combine agent answers. One answer passes through unchanged; several are merged by the LLM."""
    if llm is None:
        from src.core.config import get_llm

        llm = get_llm("synthesizer")
    llm = llm.with_retry(stop_after_attempt=max_attempts)

    def synthesizer(state: FinnieState) -> dict[str, Any]:
        outputs = state.get("agent_outputs") or []
        answered = [o for o in outputs if not o.error]
        sources = _merge_sources(answered)

        if not answered:
            return {"final_response": NO_ANSWER_MESSAGE, "sources": []}
        if len(answered) == 1:
            return {"final_response": answered[0].content, "sources": sources}

        parts = "\n\n".join(f"### Answer from the {o.agent} agent\n{o.content}" for o in answered)
        failed = [o.agent for o in outputs if o.error]
        if failed:
            parts += f"\n\n(These parts could not be answered: {', '.join(failed)}.)"
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {state.get('user_query', '')}\n\n{parts}"),
        ]
        try:
            merged = llm.invoke(messages).text.strip()
        except Exception as exc:  # noqa: BLE001 - fall back to showing each answer in turn
            logger.warning("synthesizer failed, concatenating answers: %s", exc)
            return {
                "final_response": "\n\n".join(o.content for o in answered),
                "sources": sources,
                "errors": [f"synthesizer: {exc}"],
            }
        return {"final_response": merged or "\n\n".join(o.content for o in answered), "sources": sources}

    return synthesizer


def compliance_guard(state: FinnieState) -> dict[str, Any]:
    """Add a caveat if the answer reads like personal advice, and always append the disclaimer."""
    response = state.get("final_response") or NO_ANSWER_MESSAGE
    flagged = find_advice_phrases(response)
    if flagged:
        logger.warning("session=%s advice-like phrasing: %s", state.get("session_id"), flagged)
        response = f"{response}\n\n{ADVICE_CAVEAT}"
    if not has_disclaimer(response):
        response = f"{response}\n\n---\n*{DISCLAIMER}*"
    return {"final_response": response}


def update_memory(state: FinnieState) -> dict[str, Any]:
    """Record Finnie's answer in the conversation history, without the disclaimer footer.

    Keeping the footer out of history stops the LLM from copying it into later answers.
    """
    return {"messages": [AIMessage(content=strip_disclaimer(state.get("final_response", "")))]}
