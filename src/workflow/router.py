"""Workflow router.

Classifies the user's message into one or more agents using an LLM with
structured output (RouterDecision). If the LLM call fails or returns nothing
usable, a keyword/regex classifier takes over; if that finds nothing, the
query goes to finance_qa.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from typing import get_args

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.prompts import ROUTER_SYSTEM_PROMPT
from src.core.models import AgentName, ExtractedEntities, Holding, RouterDecision
from src.workflow.state import FinnieState

logger = logging.getLogger(__name__)

ALL_AGENTS: tuple[str, ...] = get_args(AgentName)
DEFAULT_AGENT = "finance_qa"
HISTORY_TURNS = 6  # recent messages sent to the LLM so follow-ups ("what about its fees?") resolve
MAX_HISTORY_CHARS = 500

# Keyword fallback: an agent is selected when any of its patterns matches (case-insensitive).
KEYWORD_PATTERNS: dict[str, list[str]] = {
    "tax": [
        r"\btax(es|ed|able)?\b", r"\b(roth|traditional)\b", r"\bira\b", r"\b401\(?k\)?", r"\bhsa\b",
        r"\bcapital gains?\b", r"\btax[- ]loss\b", r"\bdeductib",
    ],
    "news": [r"\bnews\b", r"\bheadlines?\b", r"\bannounce", r"\bwhat happened\b", r"\blatest on\b"],
    "portfolio": [
        r"\bmy (portfolio|holdings|investments|stocks|funds)\b", r"\bi (own|have|hold) \d",
        r"\b(analy[sz]e|review) my\b", r"\bmy (allocation|asset mix)\b",
    ],
    "market": [
        r"\b(stock )?price\b", r"\bquote\b", r"\btrading at\b", r"\bperform(ing|ance)\b",
        r"\bmarket (today|now|this week)\b", r"\bs&p\b", r"\bnasdaq\b", r"\bdow\b",
        r"\bmoving average\b", r"\b52[- ]week\b", r"\bdoing (today|lately)\b",
    ],
    "goal_planning": [
        r"\bgoal\b", r"\bsav(e|ing) (up )?for\b", r"\bretire(ment)?\b", r"\bdown payment\b",
        r"\bcollege fund\b", r"\bhow much (should|do) i (need to )?(save|invest)\b",
        r"\b(per|a|each) month\b", r"\bmonthly\b",
    ],
}

# Uppercase words that look like tickers but usually aren't.
TICKER_STOPWORDS = {
    "I", "A", "AM", "AN", "AND", "ARE", "AT", "BE", "BY", "DO", "FOR", "IF", "IN", "IS", "IT", "ME",
    "MY", "NO", "OF", "OK", "ON", "OR", "SO", "THE", "TO", "UP", "US", "WE",
    "ETF", "IRA", "HSA", "USD", "CEO", "IPO", "GDP", "FAQ", "REIT", "ROI", "APR", "APY", "CD",
    "EPS", "PE", "SEC", "FDIC", "LLC", "AI", "DOW", "S", "P", "K", "YTD",
}
_TICKER_RE = re.compile(r"(?<![\w$])\$?([A-Z]{1,5})\b")
_HOLDING_RE = re.compile(r"(\d+(?:\.\d+)?)\s+(?:shares?\s+(?:of\s+)?)?\$?([A-Z]{1,5})\b")
_AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\b", re.IGNORECASE)
_HORIZON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}


def extract_entities(text: str) -> ExtractedEntities:
    """Regex-based entity extraction used by the keyword fallback."""
    tickers = [t for t in _TICKER_RE.findall(text) if t not in TICKER_STOPWORDS]
    holdings = [
        Holding(ticker=ticker, quantity=float(qty))
        for qty, ticker in _HOLDING_RE.findall(text)
        if ticker not in TICKER_STOPWORDS and float(qty) > 0
    ]
    amounts = [
        float(num.replace(",", "")) * _MULTIPLIERS.get((suffix or "").lower(), 1)
        for num, suffix in _AMOUNT_RE.findall(text)
    ]
    horizon = _HORIZON_RE.search(text)
    return ExtractedEntities(
        tickers=tickers,
        amounts=amounts,
        horizon_years=float(horizon.group(1)) if horizon else None,
        holdings=holdings,
    )


def keyword_route(query: str, available_agents: Iterable[str] = ALL_AGENTS) -> RouterDecision:
    """Deterministic fallback classifier; defaults to finance_qa when nothing matches."""
    text = query.lower()
    entities = extract_entities(query)
    agents = [
        agent
        for agent, patterns in KEYWORD_PATTERNS.items()
        if agent in available_agents and any(re.search(p, text) for p in patterns)
    ]
    if entities.holdings and "portfolio" in available_agents and "portfolio" not in agents:
        agents.append("portfolio")
    if not agents:
        return RouterDecision(
            agents=[DEFAULT_AGENT], reason="Keyword fallback: no specific intent matched.", extracted=entities
        )
    return RouterDecision(agents=agents, reason="Keyword fallback: matched " + ", ".join(agents) + ".", extracted=entities)


def _format_history(history: Sequence[BaseMessage]) -> str:
    lines = []
    for message in history[-HISTORY_TURNS:]:
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{message.type}: {content[:MAX_HISTORY_CHARS]}")
    return "\n".join(lines)


class Router:
    """LLM router with keyword fallback.

    `available_agents` restricts routing to agents that are actually implemented
    (News and Tax are stretch goals); anything else is dropped.
    """

    def __init__(self, llm: BaseChatModel, available_agents: Iterable[str] = ALL_AGENTS, max_attempts: int = 3):
        self.available_agents = tuple(a for a in available_agents if a in ALL_AGENTS)
        if DEFAULT_AGENT not in self.available_agents:
            raise ValueError(f"{DEFAULT_AGENT} must be available as the default route")
        self._chain = llm.with_structured_output(RouterDecision).with_retry(stop_after_attempt=max_attempts)

    def route(self, query: str, history: Sequence[BaseMessage] = ()) -> tuple[RouterDecision, str | None]:
        """Return the routing decision and, if the LLM path failed, an error message."""
        if not query or not query.strip():
            return RouterDecision(agents=[DEFAULT_AGENT], reason="Empty message."), None

        try:
            decision = self._chain.invoke(self._build_messages(query, history))
            if not isinstance(decision, RouterDecision):
                raise TypeError(f"router LLM returned {type(decision).__name__}, expected RouterDecision")
        except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to keyword routing
            logger.warning("LLM routing failed, using keyword fallback: %s", exc)
            return keyword_route(query, self.available_agents), f"router: LLM routing failed ({exc})"

        return self._restrict_to_available(decision, query), None

    def _build_messages(self, query: str, history: Sequence[BaseMessage]) -> list[BaseMessage]:
        prompt = ROUTER_SYSTEM_PROMPT
        unavailable = [a for a in ALL_AGENTS if a not in self.available_agents]
        if unavailable:
            prompt += "\nThese agents are currently unavailable; never select them: " + ", ".join(unavailable) + "."
        context = _format_history(history)
        user_content = f"Conversation so far:\n{context}\n\nCurrent message: {query}" if context else query
        return [SystemMessage(content=prompt), HumanMessage(content=user_content)]

    def _restrict_to_available(self, decision: RouterDecision, query: str) -> RouterDecision:
        """a safety check that runs after the LLM picks agents. It makes sure the route 
        only contains agents you've actually enabled."""
        agents = [a for a in dict.fromkeys(decision.agents) if a in self.available_agents]
        if agents == decision.agents:
            return decision
        if not agents:
            # The LLM picked only unavailable agents; let keywords try before defaulting.
            fallback = keyword_route(query, self.available_agents)
            return fallback.model_copy(update={"extracted": decision.extracted})
        return decision.model_copy(update={"agents": agents})


def make_router_node(llm: BaseChatModel | None = None, available_agents: Iterable[str] = ALL_AGENTS):
    """Build the LangGraph `router` node. Uses the router's own configured LLM unless one is given."""
    if llm is None:
        from src.core.config import get_llm

        llm = get_llm("router")
    router = Router(llm, available_agents)

    def router_node(state: FinnieState) -> dict:
        query, history = state.get("user_query", ""), list(state.get("messages", []))
        if history and isinstance(history[-1], HumanMessage) and history[-1].content == query:
            history = history[:-1]  # the current message is sent separately, not as history
        decision, error = router.route(query, history)
        logger.info("session=%s route=%s reason=%s", state.get("session_id"), decision.agents, decision.reason)
        update = {
            "route": list(decision.agents),
            "route_reason": decision.reason,
            "extracted_entities": decision.extracted,
        }
        if error:
            update["errors"] = [error]
        return update

    return router_node


def select_agents(state: FinnieState) -> list[str]:
    """Conditional-edge function: returning several node names makes LangGraph run them in parallel."""
    return state.get("route") or [DEFAULT_AGENT]
