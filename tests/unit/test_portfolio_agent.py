from datetime import UTC, datetime

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents.portfolio_agent import NO_PORTFOLIO_NOTE, PortfolioAgent, format_profile
from src.core.models import ExtractedEntities, FinancialGoal, Holding, Portfolio, Quote, UserProfile

NOW = datetime(2026, 11, 1, 14, 5, tzinfo=UTC)


class FakeMarketData:
    def __init__(self, prices: dict[str, float]):
        self.prices = prices
        self.requested = []

    def get_quotes(self, tickers):
        self.requested.append(list(tickers))
        return {t: Quote(ticker=t, price=p, change=0, change_pct=0, as_of=NOW, source="alpha_vantage")
                for t, p in self.prices.items() if t in tickers}


def agent(prices: dict[str, float] | None = None) -> tuple[PortfolioAgent, FakeMarketData]:
    market_data = FakeMarketData(prices or {})
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="Your portfolio is concentrated.")]))
    return PortfolioAgent(llm=llm, market_data=market_data), market_data


def state(**values) -> dict:
    return {"session_id": "s1", "user_query": "Analyze my portfolio", "messages": [],
            "extracted_entities": ExtractedEntities(), **values}


SAVED = Portfolio(holdings=[
    Holding(ticker="AAPL", quantity=10, cost_basis=150, asset_type="stock"),
    Holding(ticker="VTI", quantity=10, cost_basis=200, asset_type="etf"),
    Holding(ticker="CASH", quantity=1000, cost_basis=1, asset_type="cash"),
])


def test_saved_portfolio_is_priced_and_calculated_in_code():
    portfolio_agent, market_data = agent({"AAPL": 300, "VTI": 400})
    response = portfolio_agent.run(state(portfolio=SAVED))

    assert response.error is None and response.content == "Your portfolio is concentrated."
    assert market_data.requested == [["AAPL", "VTI"]]  # cash is never quoted
    assert response.data["metrics"]["total_value"] == 8000
    assert response.data["metrics"]["gain"] == 3500  # (300-150)*10 + (400-200)*10
    summary = response.data["summary"]
    assert "the user's saved portfolio" in summary
    assert "Total value: $8,000.00" in summary
    assert summary.index("- VTI") < summary.index("- AAPL") < summary.index("- CASH")  # largest first
    assert "[live quote]" in summary and "[cash]" in summary


def test_holdings_in_the_message_take_priority_and_unpriced_are_reported():
    portfolio_agent, _ = agent({"AAPL": 300})
    entities = ExtractedEntities(holdings=[Holding(ticker="AAPL", quantity=10), Holding(ticker="NOPE", quantity=5)])
    data = portfolio_agent.prefetch_data(state(portfolio=SAVED, extracted_entities=entities))

    assert "holdings named in the user's message" in data["summary"]
    assert data["unpriced"] == ["NOPE"]
    assert "Gain/loss: unknown" in data["summary"]


def test_without_a_portfolio_the_user_is_told_how_to_add_one():
    portfolio_agent, market_data = agent()
    s = state()
    data = portfolio_agent.prefetch_data(s)

    assert data is None and market_data.requested == []
    assert NO_PORTFOLIO_NOTE in portfolio_agent.build_user_message(s, [], data)


def test_no_priced_holdings():
    portfolio_agent, _ = agent()
    entities = ExtractedEntities(holdings=[Holding(ticker="NOPE", quantity=5)])
    data = portfolio_agent.prefetch_data(state(extracted_entities=entities))
    assert "No holding could be priced: NOPE" in data["summary"]


def test_prompt_includes_figures_and_the_users_goals():
    portfolio_agent, _ = agent({"AAPL": 300, "VTI": 400})
    goal = FinancialGoal(name="House", target_amount=60000, horizon_years=5, risk_tolerance="moderate")
    profile = UserProfile(session_id="s1", risk_tolerance="conservative", goals=[goal])
    s = state(portfolio=SAVED, user_profile=profile)
    message = portfolio_agent.build_user_message(s, [], portfolio_agent.prefetch_data(s))

    assert "Portfolio figures calculated in code" in message
    assert "risk tolerance: conservative" in message and "House, $60,000.00 in 5 years" in message
    assert message.endswith("Question: Analyze my portfolio")
    assert "unknown" in format_profile(None)
