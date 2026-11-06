"""Goal Planning Agent: projects the user's savings goals and explains them.

Which goal is projected:
1. a saved goal named in the message ("my house goal"), or the only saved goal when the
   message asks about "my goal" or being "on track";
2. every saved goal when the message asks about "my goals";
3. otherwise none: the LLM calls `project_goal` with numbers from the message, or asks for them.

Human in the loop: when a saved goal is projected, the user has a portfolio, and they
haven't said how much of it counts toward that goal, the node pauses the graph with
`interrupt()` to ask. The answer is added to the goal's current savings for the rest of
the session (or saved to the goal by the UI). On resume LangGraph re-runs the node from
the start, so the question is asked before any retrieval or LLM call.
"""

import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from src.agents.base_agent import BaseAgent
from src.agents.prompts import GOAL_PLANNING_SYSTEM_PROMPT
from src.core.goal_projection import ASSUMED_RETURNS, format_projection, project_goal
from src.core.models import FinancialGoal, Portfolio, RetrievedDocument, UserProfile
from src.core.portfolio_metrics import price_holdings
from src.data.market_data import MarketDataService
from src.workflow.state import FinnieState

NO_GOAL_NOTE = (
    "No saved goal was selected for this question. If the user's message gives a target amount and a "
    "time frame, call project_goal with those numbers (current savings and monthly contribution are 0 "
    "if not given) and the user's risk tolerance, then explain the result. If the target or time frame "
    "is missing, don't invent numbers: ask for the target amount, the timeline, what they have saved "
    "and what they can save each month."
)
SINGLE_GOAL_RE = re.compile(r"\b(my goal|on track|my savings? goal|my plan)\b", re.IGNORECASE)
ALL_GOALS_RE = re.compile(r"\b(my goals|all (of )?my goals|each goal|every goal)\b", re.IGNORECASE)
NAME_STOPWORDS = {"goal", "fund", "savings", "saving", "save", "plan", "money", "the", "for", "and", "new", "my"}


def _name_matches(goal: FinancialGoal, query: str) -> bool:
    """A goal matches when a distinctive word of its name, or its first six letters, is in the query."""
    words = [w for w in re.findall(r"[a-z0-9]+", goal.name.lower()) if len(w) >= 3 and w not in NAME_STOPWORDS]
    return any(w[:6] in query for w in words)


def goals_in_question(profile: UserProfile | None, query: str) -> list[FinancialGoal]:
    """The saved goals the message is about (see the module docstring)."""
    goals = profile.goals if profile else []
    if not goals:
        return []
    named = [g for g in goals if _name_matches(g, query.lower())]
    if named:
        return named
    if ALL_GOALS_RE.search(query):
        return list(goals)
    if len(goals) == 1 and SINGLE_GOAL_RE.search(query):
        return list(goals)
    return []


def portfolio_amount(goal: FinancialGoal, answers: dict[str, float]) -> float:
    """Portfolio dollars counted toward the goal: the saved choice, else this session's answer, else 0."""
    if goal.from_portfolio is not None:
        return goal.from_portfolio
    return answers.get(goal.name, 0.0)


def _parse_amount(answer: Any, limit: float) -> float:
    """The amount from a resume value ({"amount": ...}), clamped to [0, limit]; anything else counts as 0."""
    try:
        amount = float(answer.get("amount", 0)) if isinstance(answer, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0
    return min(max(amount, 0.0), limit)


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    description = "Saving for a goal, projections and monthly contributions"
    system_prompt = GOAL_PLANNING_SYSTEM_PROMPT
    retrieval_k = 3
    retrieval_categories = ["goal_planning", "basics", "retirement"]  # noqa: RUF012 - read-only

    def __init__(self, market_data: MarketDataService | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.market_data = market_data or MarketDataService.from_settings()

    # ---- human in the loop -----------------------------------------------

    def portfolio_value(self, portfolio: Portfolio | None) -> float:
        if portfolio is None:
            return 0.0
        tickers = [h.ticker for h in portfolio.holdings if h.asset_type != "cash"]
        rows, _ = price_holdings(portfolio, self.market_data.get_quotes(tickers))
        return sum(row["quantity"] * row["price"] for row in rows)

    def question_for_user(self, state: FinnieState) -> dict[str, Any] | None:
        """The interrupt payload when a portfolio amount is needed for the goal in question, else None."""
        goals = goals_in_question(state.get("user_profile"), state.get("user_query", ""))
        answers = state.get("goal_portfolio_answers") or {}
        if len(goals) != 1 or state.get("portfolio") is None:
            return None
        goal = goals[0]
        if goal.from_portfolio is not None or goal.name in answers:
            return None
        value = self.portfolio_value(state["portfolio"])
        if value <= 0:
            return None
        return {
            "kind": "goal_portfolio_amount",
            "goal": goal.name,
            "goal_savings": goal.current_savings,
            "portfolio_value": round(value, 2),
            "question": (f"Before I project this: your portfolio is currently worth about **${value:,.0f}**, and your "
                         f"*{goal.name}* goal shows **${goal.current_savings:,.0f}** already saved. How much of your "
                         "portfolio should count toward this goal?"),
        }

    def as_node(self):
        def node(state: FinnieState) -> dict:
            answers = dict(state.get("goal_portfolio_answers") or {})
            question = self.question_for_user(state)
            if question is not None:
                answer = interrupt(question)  # pauses here; returns the user's answer on resume
                answers[question["goal"]] = _parse_amount(answer, question["portfolio_value"])
                state = {**state, "goal_portfolio_answers": answers}
            response = self.run(state)
            update: dict[str, Any] = {"agent_outputs": [response], "goal_portfolio_answers": answers}
            if response.error:
                update["errors"] = [f"{self.name}: {response.error}"]
            return update

        node.__name__ = self.name
        return node

    # ---- agent steps -----------------------------------------------------

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        return self.retrieve(state.get("user_query", ""), categories=self.retrieval_categories)

    def prefetch_data(self, state: FinnieState) -> dict[str, Any] | None:
        goals = goals_in_question(state.get("user_profile"), state.get("user_query", ""))
        if not goals:
            return None
        answers = state.get("goal_portfolio_answers") or {}
        projections = [
            project_goal(g.name, g.target_amount, g.horizon_years, g.risk_tolerance, g.current_savings,
                         g.monthly_contribution, portfolio_amount(g, answers))
            for g in goals
        ]
        return {
            "summary": "\n\n".join(format_projection(p) for p in projections),
            "projections": [p.model_dump() for p in projections],
        }

    def get_tools(self) -> list[BaseTool]:
        def project_goal_tool(target_amount: float, horizon_years: float, current_savings: float = 0.0,
                              monthly_contribution: float = 0.0, risk_tolerance: str = "moderate") -> str:
            """Project a savings goal: future value with a low/high range, the monthly amount needed and
            the time to reach the target. risk_tolerance is conservative, moderate or aggressive."""
            risk = risk_tolerance.strip().lower()
            projection = project_goal("Goal from the user's message", target_amount, horizon_years,
                                      risk if risk in ASSUMED_RETURNS else "moderate",
                                      current_savings, monthly_contribution)
            return format_projection(projection)

        return [StructuredTool.from_function(project_goal_tool, name="project_goal")]

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        profile = state.get("user_profile")
        parts = [f"Goal projections calculated in code:\n\n{data['summary']}" if data else NO_GOAL_NOTE]
        about = [f"The user's risk tolerance: {(profile.risk_tolerance if profile else None) or 'moderate'}."]
        if profile and profile.goals:
            about.append("Saved goals: " + ", ".join(g.name for g in profile.goals) + ".")
        parts.append(" ".join(about))
        if docs:
            parts.append("Knowledge base excerpts for explaining concepts (cite them by number, e.g. [1]):\n\n"
                         + self.format_context(docs))
        parts.append(f"Question: {state.get('user_query', '')}")
        return "\n\n".join(parts)
