"""Market Analysis Agent: how specific US tickers are doing, from live or cached market data.

The tickers the router extracted are pre-fetched in code (a quote and a trend summary
each), so "How is AAPL doing?" needs a single LLM call. The LLM also gets tools for a
ticker the router missed, such as one only named earlier in the conversation. Concepts
like moving averages or volatility are explained from the knowledge base.
"""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from src.agents.base_agent import BaseAgent
from src.agents.prompts import MARKET_SYSTEM_PROMPT
from src.core.models import Quote, RetrievedDocument, TrendSummary
from src.data.market_data import TICKER_PATTERN, MarketDataService
from src.workflow.state import FinnieState

MAX_TICKERS = 3  # each ticker costs up to 2 requests of the free tier's ~25 a day
MARKET_PROXY = "SPY"  # stands in for "the market" (the S&P 500)

SOURCE_LABELS = {"alpha_vantage": "live from Alpha Vantage", "cache": "cached from Alpha Vantage", "mock": "sample data"}

NO_TICKER_NOTE = (
    "No ticker was identified in the message. If the conversation makes clear which ticker the "
    f"user means, call the tools for it. If they ask about the market as a whole, use {MARKET_PROXY} "
    "(an ETF that tracks the S&P 500) and say so. Otherwise ask which stock or fund they mean."
)


def _pct(value: float | None) -> str:
    return "n/a (not enough history)" if value is None else f"{value:+.1%}"


def _money(value: float | None) -> str:
    return "n/a (not enough history)" if value is None else f"${value:,.2f}"


def format_market_data(ticker: str, quote: Quote | None, trend: TrendSummary | None) -> str:
    """A plain-text block for the prompt, with percentages already formatted so the LLM doesn't convert them."""
    if quote is None and trend is None:
        return f"{ticker}: no data available (unknown ticker, or the data provider can't be reached right now)."
    lines = [ticker]
    if quote is not None:
        lines.append(
            f"- Latest quote: ${quote.price:,.2f}, {quote.change:+,.2f} ({quote.change_pct / 100:+.2%}) on the day "
            f"[{SOURCE_LABELS[quote.source]}, retrieved {quote.as_of:%Y-%m-%d %H:%M} UTC]"
        )
    if trend is not None:
        lines.append(f"- Weekly closes up to {trend.last_date:%Y-%m-%d}: last close {_money(trend.last_close)} "
                     f"[{SOURCE_LABELS[trend.source]}, retrieved {trend.as_of:%Y-%m-%d %H:%M} UTC]")
        lines.append(f"- Change: 1 month {_pct(trend.change_1m)}, 6 months {_pct(trend.change_6m)}, "
                     f"1 year {_pct(trend.change_1y)}")
        lines.append(f"- 10-week moving average (≈ 50-day) {_money(trend.sma_10w)}; "
                     f"40-week moving average (≈ 200-day) {_money(trend.sma_40w)}")
        lines.append(f"- Volatility (annualized, from the last year's weekly returns): "
                     f"{_pct(trend.volatility).lstrip('+')}")
    return "\n".join(lines)


class MarketAgent(BaseAgent):
    name = "market"
    description = "Current prices, quotes and trends for specific tickers or indexes"
    system_prompt = MARKET_SYSTEM_PROMPT
    retrieval_k = 3
    retrieval_categories = ["market_concepts", "risk", "stocks", "etfs"]  # noqa: RUF012 - read-only

    def __init__(self, market_data: MarketDataService | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.market_data = market_data or MarketDataService.from_settings()

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        return self.retrieve(state.get("user_query", ""), categories=self.retrieval_categories)

    def tickers(self, state: FinnieState) -> list[str]:
        entities = state.get("extracted_entities")
        tickers = [t for t in (entities.tickers if entities else []) if TICKER_PATTERN.match(t)]
        return tickers[:MAX_TICKERS]

    def prefetch_data(self, state: FinnieState) -> dict[str, Any] | None:
        tickers = self.tickers(state)
        if not tickers:
            return None
        quotes = {t: self.market_data.get_quote(t) for t in tickers}
        trends = {t: self.market_data.get_trend(t) for t in tickers}
        return {
            "tickers": tickers,
            "quotes": {t: q.model_dump(mode="json") for t, q in quotes.items() if q},
            "trends": {t: s.model_dump(mode="json") for t, s in trends.items() if s},
            "missing": [t for t in tickers if quotes[t] is None and trends[t] is None],
            "summary": "\n\n".join(format_market_data(t, quotes[t], trends[t]) for t in tickers),
        }

    def get_tools(self) -> list[BaseTool]:
        def get_quote(ticker: str) -> str:
            """Latest price and today's change for a US-listed stock or ETF ticker, e.g. AAPL."""
            ticker = ticker.strip().upper()
            return format_market_data(ticker, self.market_data.get_quote(ticker), None)

        def get_price_trend(ticker: str) -> str:
            """1-month, 6-month and 1-year change, moving averages and volatility for a US ticker."""
            ticker = ticker.strip().upper()
            return format_market_data(ticker, None, self.market_data.get_trend(ticker))

        return [StructuredTool.from_function(get_quote), StructuredTool.from_function(get_price_trend)]

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        parts = [f"Market data:\n\n{data['summary']}" if data else NO_TICKER_NOTE]
        if docs:
            parts.append(
                "Knowledge base excerpts for explaining concepts (cite them by number, e.g. [1]):\n\n"
                + self.format_context(docs)
            )
        parts.append(f"Question: {state.get('user_query', '')}")
        return "\n\n".join(parts)
