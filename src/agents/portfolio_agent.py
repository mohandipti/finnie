"""Portfolio Analysis Agent: explains the user's holdings using metrics calculated in code.

Which holdings are analyzed, in order of preference:
1. holdings named in the message ("I have 10 AAPL and 5 VTI");
2. the user's saved portfolio.
With neither, the agent explains how to add holdings instead.

Holdings are priced at the latest quotes (cash at $1, the price paid as a fallback), then
compute_metrics works out value, allocation, gain/loss, fees, diversification and risk.
The LLM gets those figures verbatim and only explains them.
"""

import json
from typing import Any

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.prompts import PORTFOLIO_SYSTEM_PROMPT
from src.core.models import Portfolio, RetrievedDocument, UserProfile
from src.core.portfolio_metrics import RISK_WEIGHTS, compute_metrics, price_holdings
from src.data.market_data import MarketDataService
from src.workflow.state import FinnieState

ASSET_TYPE_LABELS = {"stock": "stocks", "etf": "ETFs", "bond": "bonds", "cash": "cash",
                     "mutual_fund": "mutual funds", "other": "other or unknown type"}
PRICE_LABELS = {"alpha_vantage": "live quote", "cache": "cached quote", "cost_basis": "price paid (no quote)",
                "cash": "cash"}
NO_PORTFOLIO_NOTE = (
    "The user hasn't saved a portfolio and didn't list any holdings, so there is nothing to analyze. "
    "Say so, and explain that they can add their holdings and cash on the My account page, or list them "
    "in the message (for example \"I have 10 AAPL and 5 VTI\"). You may briefly explain what a portfolio "
    "analysis covers (allocation, diversification, fees and risk), without inventing any figures."
)
RISK_SCALE = ", ".join(f"{ASSET_TYPE_LABELS[t]} {w}" for t, w in RISK_WEIGHTS.items())


def _money(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def format_analysis(df: pd.DataFrame, metrics: dict[str, Any], label: str, unpriced: list[str]) -> str:
    """The calculated figures as plain text for the prompt, largest holding first."""
    lines = [f"Portfolio analyzed: {label}", f"Total value: {_money(metrics['total_value'])}"]
    if metrics["gain"] is None:
        lines.append("Gain/loss: unknown (no price paid was given)")
    else:
        lines.append(f"Gain/loss on holdings with a known price paid: {_money(metrics['gain'])} "
                     f"({metrics['gain_pct']:+.1%})")
    if metrics["expense_ratio"] is not None:
        lines.append(f"Weighted average expense ratio: {metrics['expense_ratio']:.2%} a year")
    lines.append(f"Diversification score: {metrics['diversification']:.0f}/100 "
                 "(100 × (1 − sum of squared holding weights); 0 means everything is in one holding)")
    lines.append(f"Risk level: {metrics['risk_level']} (score {metrics['risk_score']:.1f} on a 1-5 scale, "
                 f"the value-weighted average of: {RISK_SCALE})")
    by_type = df.groupby("asset_type")["weight"].sum().sort_values(ascending=False)
    lines.append("Allocation by asset type: " + ", ".join(
        f"{ASSET_TYPE_LABELS.get(t, t)} {w:.1%}" for t, w in by_type.items()))
    lines.append("Holdings, largest first:")
    for row in df.sort_values("value", ascending=False).itertuples():
        line = (f"- {row.ticker} ({ASSET_TYPE_LABELS.get(row.asset_type, row.asset_type)}): "
                f"{row.quantity:g} × {_money(row.price)} = {_money(row.value)}, {row.weight:.1%} of the portfolio")
        if pd.notna(row.gain):
            line += f", gain/loss {_money(row.gain)}"
        if pd.notna(row.expense_ratio) and row.expense_ratio > 0:
            line += f", expense ratio {row.expense_ratio:.2%}"
        lines.append(line + f" [{PRICE_LABELS.get(row.price_source, row.price_source)}]")
    if unpriced:
        lines.append(f"Left out because no price is available: {', '.join(unpriced)}")
    return "\n".join(lines)


def format_profile(profile: UserProfile | None) -> str:
    if profile is None:
        return "The user's risk tolerance and goals are unknown."
    lines = [f"The user's stated risk tolerance: {profile.risk_tolerance or 'not given'}."]
    for goal in profile.goals:
        lines.append(f"Goal: {goal.name}, {_money(goal.target_amount)} in {goal.horizon_years:g} years "
                     f"({goal.risk_tolerance} risk).")
    return "\n".join(lines)


class PortfolioAgent(BaseAgent):
    name = "portfolio"
    description = "Analyzing the user's own holdings, allocation and diversification"
    system_prompt = PORTFOLIO_SYSTEM_PROMPT
    retrieval_k = 3
    retrieval_categories = ["diversification", "risk", "mutual_funds", "etfs"]  # noqa: RUF012 - read-only

    def __init__(self, market_data: MarketDataService | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.market_data = market_data or MarketDataService.from_settings()

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        return self.retrieve(state.get("user_query", ""), categories=self.retrieval_categories)

    def portfolio_to_analyze(self, state: FinnieState) -> tuple[Portfolio, str] | None:
        """The portfolio and a label saying whose it is, or None if the user has none."""
        entities = state.get("extracted_entities")
        if entities and entities.holdings:
            return Portfolio(holdings=entities.holdings), "the holdings named in the user's message"
        if state.get("portfolio") is not None:
            return state["portfolio"], "the user's saved portfolio"
        return None

    def prefetch_data(self, state: FinnieState) -> dict[str, Any] | None:
        found = self.portfolio_to_analyze(state)
        if found is None:
            return None
        portfolio, label = found
        tickers = [h.ticker for h in portfolio.holdings if h.asset_type != "cash"]
        rows, unpriced = price_holdings(portfolio, self.market_data.get_quotes(tickers))
        if not rows:
            return {"summary": f"Portfolio analyzed: {label}\nNo holding could be priced: {', '.join(unpriced)}",
                    "unpriced": unpriced}
        df, metrics = compute_metrics(rows)
        return {
            "summary": format_analysis(df, metrics, label, unpriced),
            "metrics": {k: float(v) if isinstance(v, int | float) else v for k, v in metrics.items()},
            "holdings": json.loads(df.to_json(orient="records")),  # plain JSON types for the checkpointer
            "unpriced": unpriced,
        }

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        figures = f"Portfolio figures calculated in code:\n\n{data['summary']}" if data else NO_PORTFOLIO_NOTE
        parts = [figures, format_profile(state.get("user_profile"))]
        if docs:
            parts.append("Knowledge base excerpts for explaining concepts (cite them by number, e.g. [1]):\n\n"
                         + self.format_context(docs))
        parts.append(f"Question: {state.get('user_query', '')}")
        return "\n\n".join(parts)
