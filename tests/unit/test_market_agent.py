from datetime import UTC, date, datetime

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents.market_agent import NO_TICKER_NOTE, MarketAgent, format_market_data
from src.core.models import ExtractedEntities, Quote, TrendSummary
from src.workflow.router import keyword_route

NOW = datetime(2026, 11, 1, 14, 5, tzinfo=UTC)
AAPL_QUOTE = Quote(ticker="AAPL", price=250.5, change=-2.5, change_pct=-0.9881, as_of=NOW, source="alpha_vantage")
AAPL_TREND = TrendSummary(ticker="AAPL", last_close=251.0, last_date=date(2026, 10, 30), change_1m=0.021,
                          change_6m=-0.05, change_1y=0.124, sma_10w=245.0, sma_40w=230.0, volatility=0.25,
                          as_of=NOW, source="cache")


class FakeMarketData:
    def __init__(self):
        self.calls = []

    def get_quote(self, ticker):
        self.calls.append(("quote", ticker))
        return AAPL_QUOTE if ticker == "AAPL" else None

    def get_trend(self, ticker):
        self.calls.append(("trend", ticker))
        return AAPL_TREND if ticker == "AAPL" else None


class FakeLLM(GenericFakeChatModel):
    """Replies with the given messages in order; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def agent(*replies: AIMessage) -> tuple[MarketAgent, FakeMarketData, FakeLLM]:
    llm = FakeLLM(messages=iter(replies))
    market_data = FakeMarketData()
    return MarketAgent(llm=llm, market_data=market_data), market_data, llm


def state(query: str, tickers: list[str]) -> dict:
    return {"session_id": "s1", "user_query": query, "messages": [],
            "extracted_entities": ExtractedEntities(tickers=tickers)}


def test_prefetches_quote_and_trend_for_extracted_tickers():
    market, market_data, _ = agent(AIMessage(content="AAPL closed at $250.50."))
    response = market.run(state("How is AAPL doing?", ["AAPL", "NOPE"]))

    assert response.error is None and response.agent == "market"
    assert response.content == "AAPL closed at $250.50."
    assert set(response.data["quotes"]) == {"AAPL"}
    assert response.data["missing"] == ["NOPE"]
    assert ("quote", "AAPL") in market_data.calls and ("trend", "NOPE") in market_data.calls


def test_tickers_are_capped_and_invalid_ones_dropped():
    market, _, _ = agent()
    tickers = market.tickers(state("compare", ["S&P", "AAPL", "MSFT", "BRK.B", "VTI"]))
    assert tickers == ["AAPL", "MSFT", "BRK.B"]


def test_without_a_ticker_the_llm_can_call_a_tool():
    tool_call = AIMessage(content="", tool_calls=[{"name": "get_price_trend", "args": {"ticker": "aapl"}, "id": "1"}])
    market, market_data, _ = agent(tool_call, AIMessage(content="Over the past year AAPL rose 12.4%."))
    response = market.run(state("What about its trend?", []))

    assert response.content == "Over the past year AAPL rose 12.4%."
    assert market_data.calls == [("trend", "AAPL")]
    assert "+12.4%" in response.data["tool_results"][0]["result"]


def test_prompt_includes_formatted_data_or_a_no_ticker_note():
    market, _, _ = agent()
    s = state("How is AAPL doing?", ["AAPL"])
    message = market.build_user_message(s, [], market.prefetch_data(s))
    assert "Market data:" in message and "Question: How is AAPL doing?" in message

    assert NO_TICKER_NOTE in market.build_user_message(state("How's it doing?", []), [], None)


def test_format_market_data():
    text = format_market_data("AAPL", AAPL_QUOTE, AAPL_TREND)
    assert "$250.50, -2.50 (-0.99%) on the day [live from Alpha Vantage, retrieved 2026-11-01 14:05 UTC]" in text
    assert "1 month +2.1%, 6 months -5.0%, 1 year +12.4%" in text
    assert "≈ 200-day) $230.00" in text and "Volatility" in text and "25.0%" in text
    assert "no data available" in format_market_data("NOPE", None, None)


def test_keyword_fallback_routes_ticker_questions_to_market():
    assert keyword_route("How is AAPL doing?").agents == ["market"]
    assert keyword_route("What is an ETF?").agents == ["finance_qa"]
    assert keyword_route("How is AAPL doing?", ["finance_qa"]).agents == ["finance_qa"]
