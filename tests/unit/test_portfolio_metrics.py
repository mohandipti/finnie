from datetime import UTC, datetime

import pytest

from src.core.models import Holding, Portfolio, Quote
from src.core.portfolio_metrics import compute_metrics, price_holdings
from src.data.sample_data import load_sample_portfolio


def quote(ticker: str, price: float, source: str = "alpha_vantage") -> Quote:
    return Quote(ticker=ticker, price=price, change=0, change_pct=1.5, as_of=datetime.now(UTC), source=source)


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(holdings=[
        Holding(ticker="VTI", quantity=10, cost_basis=300, asset_type="etf"),
        Holding(ticker="AAPL", quantity=5, cost_basis=150, asset_type="stock"),
        Holding(ticker="XYZ", quantity=3, asset_type="stock"),
        Holding(ticker="CASH", quantity=1000, cost_basis=1, asset_type="cash"),
    ])


def test_price_holdings_uses_quote_then_price_paid(portfolio):
    rows, unpriced = price_holdings(portfolio, {"VTI": quote("VTI", 370)})
    prices = {r["ticker"]: (r["price"], r["price_source"]) for r in rows}
    assert prices == {"VTI": (370, "alpha_vantage"), "AAPL": (150, "cost_basis"), "CASH": (1.0, "cash")}
    assert unpriced == ["XYZ"]


def test_metrics_for_a_user_portfolio(portfolio):
    rows, _ = price_holdings(portfolio, {"VTI": quote("VTI", 370), "XYZ": quote("XYZ", 20, "cache")})
    df, m = compute_metrics(rows)

    assert m["total_value"] == 3700 + 750 + 60 + 1000
    # XYZ has no price paid, so it counts towards value but not gain.
    assert m["gain"] == pytest.approx(700)
    assert m["gain_pct"] == pytest.approx(700 / (3000 + 750 + 1000))
    assert m["expense_ratio"] is None
    assert df.loc[df["ticker"] == "XYZ", "name"].item() == "XYZ"


def test_metrics_without_any_price_paid():
    rows, _ = price_holdings(Portfolio(holdings=[Holding(ticker="VTI", quantity=1)]), {"VTI": quote("VTI", 100)})
    _, m = compute_metrics(rows)
    assert (m["gain"], m["gain_pct"]) == (None, None)
    assert m["risk_level"] == "High"  # unknown asset type counts as "other"


def test_sample_portfolio_still_has_fees():
    _, m = compute_metrics(load_sample_portfolio()["holdings"])
    assert m["total_value"] == pytest.approx(29525)
    assert m["expense_ratio"] is not None
