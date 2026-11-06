from datetime import UTC, date, datetime, timedelta

import pytest

from src.core.models import Holding, Portfolio, PriceHistory, PricePoint
from src.web_app.tabs.market import history_frame, period_stats, ticker_options


def history(closes: list[float]) -> PriceHistory:
    start = date(2025, 1, 3)
    return PriceHistory(ticker="SPY", as_of=datetime.now(UTC), source="alpha_vantage",
                        points=[PricePoint(date=start + timedelta(weeks=i), close=c) for i, c in enumerate(closes)])


def test_history_frame_adds_moving_averages():
    df = history_frame(history([float(i) for i in range(1, 61)]))
    assert len(df) == 60
    assert df["sma10"].iloc[-1] == pytest.approx(sum(range(51, 61)) / 10)
    assert df["sma40"].isna().iloc[38] and not df["sma40"].isna().iloc[39]


def test_period_stats():
    df = history_frame(history([100.0] * 40 + [110.0, 121.0]))
    stats = period_stats(df, weeks=2)
    assert stats["change"] == pytest.approx(0.21)
    assert stats["volatility"] == pytest.approx(0)  # two identical 10% weekly gains
    assert stats["above_sma40"] is True


def test_period_stats_with_short_history():
    stats = period_stats(history_frame(history([100.0])), weeks=52)
    assert stats == {"change": None, "volatility": None, "above_sma40": None}


def test_ticker_options_put_holdings_first_without_cash():
    portfolio = Portfolio(holdings=[Holding(ticker="AAPL", quantity=1), Holding(ticker="NVDA", quantity=2),
                                    Holding(ticker="CASH", quantity=500, asset_type="cash")])
    assert ticker_options(portfolio)[:3] == ["AAPL", "NVDA", "SPY"]
    assert "CASH" not in ticker_options(portfolio)
    assert ticker_options(None)[0] == "SPY"
