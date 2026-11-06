from datetime import UTC, date, datetime, timedelta

import diskcache
import pytest
import requests

from src.core.models import PriceHistory, PricePoint
from src.data import market_data
from src.data.market_data import MarketDataService, summarize_trend

VTI_QUOTE = {"Global Quote": {"01. symbol": "VTI", "05. price": "371.5700", "09. change": "-6.6700",
                              "10. change percent": "-1.7634%"}}
RATE_LIMITED = {"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params["symbol"])
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(market_data, "MIN_SECONDS_BETWEEN_CALLS", 0)


def service(tmp_path, session, api_key="key", ttl=1800) -> MarketDataService:
    return MarketDataService(api_key=api_key, cache=diskcache.Cache(tmp_path), quote_ttl_seconds=ttl,
                             history_ttl_seconds=ttl, session=session)


def test_live_quote_is_parsed_and_then_served_from_cache(tmp_path):
    session = FakeSession(VTI_QUOTE)
    quotes = service(tmp_path, session)

    live = quotes.get_quote("vti")
    assert (live.ticker, live.price, live.change_pct, live.source) == ("VTI", 371.57, -1.7634, "alpha_vantage")

    cached = quotes.get_quote("VTI")
    assert (cached.price, cached.source) == (371.57, "cache")
    assert session.calls == ["VTI"]


@pytest.mark.parametrize("failure", [RATE_LIMITED, {"Global Quote": {}}, requests.ConnectionError("down")])
def test_failed_fetch_falls_back_to_last_good_quote(tmp_path, failure):
    session = FakeSession(VTI_QUOTE, failure)
    quotes = service(tmp_path, session, ttl=0.01)
    quotes.get_quote("VTI")
    quotes.cache.delete("quote:VTI")  # the fresh entry has expired

    fallback = quotes.get_quote("VTI")
    assert (fallback.price, fallback.source) == (371.57, "cache")
    assert session.calls == ["VTI", "VTI"]


def test_no_quote_without_key_or_cache(tmp_path):
    session = FakeSession()
    assert service(tmp_path, session, api_key=None).get_quote("VTI") is None
    assert session.calls == []


def test_get_quotes_skips_unknown_and_duplicate_tickers(tmp_path):
    session = FakeSession(VTI_QUOTE, {"Global Quote": {}})
    quotes = service(tmp_path, session).get_quotes(["VTI", "vti", "NOPE"])
    assert list(quotes) == ["VTI"]
    assert session.calls == ["VTI", "NOPE"]


def weekly_payload(closes: list[float]) -> dict:
    end = date.today()
    series = {(end - timedelta(weeks=len(closes) - 1 - i)).isoformat(): {"5. adjusted close": str(c)}
              for i, c in enumerate(closes)}
    return {"Meta Data": {}, "Weekly Adjusted Time Series": series}


def test_history_is_parsed_oldest_first_and_cached(tmp_path):
    session = FakeSession(weekly_payload([100, 101, 102]))
    data = service(tmp_path, session)

    history = data.get_history("spy")
    assert [p.close for p in history.points] == [100, 101, 102]
    assert history.points[0].date < history.points[-1].date
    assert data.get_history("SPY").source == "cache"
    assert session.calls == ["SPY"]


def test_history_keeps_only_recent_years(tmp_path):
    payload = weekly_payload([50.0] * 600)  # ~11.5 years of weeks
    history = service(tmp_path, FakeSession(payload)).get_history("SPY")
    assert len(history.points) <= market_data.HISTORY_YEARS * 53


def test_history_falls_back_to_last_good_copy(tmp_path):
    session = FakeSession(weekly_payload([100, 110]), RATE_LIMITED)
    data = service(tmp_path, session)
    data.get_history("SPY")
    data.cache.delete("history:SPY")
    assert [p.close for p in data.get_history("SPY").points] == [100, 110]


def test_summarize_trend():
    start = datetime(2025, 1, 3).date()
    closes = [100 + i for i in range(60)]  # steady rise over 60 weeks
    history = PriceHistory(ticker="SPY", as_of=datetime.now(UTC), source="alpha_vantage",
                           points=[PricePoint(date=start + timedelta(weeks=i), close=c) for i, c in enumerate(closes)])
    trend = summarize_trend(history)

    assert trend.last_close == 159
    assert trend.change_1m == pytest.approx(159 / 155 - 1)
    assert trend.change_1y == pytest.approx(159 / 107 - 1)
    assert trend.sma_10w == pytest.approx(sum(closes[-10:]) / 10)
    assert trend.sma_40w == pytest.approx(sum(closes[-40:]) / 40)
    assert trend.volatility is not None and trend.volatility > 0


def test_summarize_trend_with_short_history():
    history = PriceHistory(ticker="NEW", as_of=datetime.now(UTC), source="alpha_vantage",
                           points=[PricePoint(date=date(2026, 10, 30), close=10)])
    trend = summarize_trend(history)
    assert (trend.change_1m, trend.change_1y, trend.sma_40w, trend.volatility) == (None, None, None, None)
