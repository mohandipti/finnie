"""Stock quotes and price history from Alpha Vantage, cached on disk.

Alpha Vantage's free tier allows about 25 requests a day and 1 per second, so:

- a quote is reused for QUOTE_CACHE_MINUTES (30 by default), and a price history for
  HISTORY_CACHE_HOURS (12) before it is fetched again;
- the last good copy of each is kept indefinitely, and returned (marked "cache") when a
  fetch fails or the daily limit is reached;
- live requests are spaced at least a second apart.

History is weekly: full daily history is a premium feature, while weekly adjusted history is
free and goes back decades in one request. Moving averages use the weekly equivalents of the
usual daily ones (10 weeks ≈ 50 days, 40 weeks ≈ 200 days).
"""

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any, TypeVar

import diskcache
import requests

from src.core.config import Settings, get_settings
from src.core.models import PriceHistory, PricePoint, Quote, TrendSummary

logger = logging.getLogger(__name__)

API_URL = "https://www.alphavantage.co/query"
MIN_SECONDS_BETWEEN_CALLS = 1.5  # the free tier rejects bursts even at ~1.1 s apart
HISTORY_YEARS = 5  # weeks kept per ticker: enough for 1-year trends and charts
WEEKS_PER_YEAR = 52
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")  # US tickers, e.g. AAPL, BRK.B

Cached = TypeVar("Cached", Quote, PriceHistory)


class MarketDataError(RuntimeError):
    """Data couldn't be fetched (network error, unknown ticker or rate limit)."""


class MarketDataService:
    def __init__(self, api_key: str | None, cache: diskcache.Cache, quote_ttl_seconds: int = 1800,
                 history_ttl_seconds: int = 12 * 3600, session: requests.Session | None = None,
                 timeout: float = 15) -> None:
        self.api_key = api_key
        self.cache = cache
        self.quote_ttl_seconds = quote_ttl_seconds
        self.history_ttl_seconds = history_ttl_seconds
        self.session = session or requests.Session()
        self.timeout = timeout
        self._last_call = 0.0
        self._throttle = threading.Lock()  # agents running in parallel share one service

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "MarketDataService":
        settings = settings or get_settings()
        key = settings.alpha_vantage_api_key.get_secret_value() if settings.alpha_vantage_api_key else None
        return cls(api_key=key or None, cache=diskcache.Cache(settings.market_cache_dir),
                   quote_ttl_seconds=settings.quote_cache_minutes * 60,
                   history_ttl_seconds=settings.history_cache_hours * 3600)

    # Quotes

    def get_quote(self, ticker: str) -> Quote | None:
        """The latest quote for a ticker, or None if there is neither a live nor a cached one."""
        ticker = ticker.strip().upper()
        return self._cached("quote", ticker, Quote, self.quote_ttl_seconds, lambda: self._fetch_quote(ticker))

    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        quotes = {}
        for ticker in dict.fromkeys(t.strip().upper() for t in tickers):
            quote = self.get_quote(ticker)
            if quote is not None:
                quotes[ticker] = quote
        return quotes

    # Price history

    def get_history(self, ticker: str) -> PriceHistory | None:
        """Weekly adjusted closes for the last few years, or None if there is no live or cached copy."""
        ticker = ticker.strip().upper()
        return self._cached("history", ticker, PriceHistory, self.history_ttl_seconds,
                            lambda: self._fetch_history(ticker))

    def get_trend(self, ticker: str) -> TrendSummary | None:
        history = self.get_history(ticker)
        return summarize_trend(history) if history else None

    # Caching and requests

    def _cached(self, kind: str, ticker: str, model: type[Cached], ttl: int,
                fetch: Callable[[], Cached]) -> Cached | None:
        fresh = self.cache.get(f"{kind}:{ticker}")
        if fresh is not None:
            return model.model_validate(fresh).model_copy(update={"source": "cache"})
        if self.api_key:
            try:
                item = fetch()
            except MarketDataError as exc:
                logger.warning("%s for %s unavailable: %s", kind, ticker, exc)
            else:
                data = item.model_dump(mode="json")
                self.cache.set(f"{kind}:{ticker}", data, expire=ttl)
                self.cache.set(f"last-{kind}:{ticker}", data)
                return item
        last = self.cache.get(f"last-{kind}:{ticker}")
        return model.model_validate(last).model_copy(update={"source": "cache"}) if last else None

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        with self._throttle:
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
        try:
            response = self.session.get(API_URL, params={**params, "apikey": self.api_key}, timeout=self.timeout)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MarketDataError(type(exc).__name__) from exc
        # Rate limits and bad requests come back as 200 with a message instead of data.
        for key in ("Note", "Information", "Error Message"):
            if key in payload:
                raise MarketDataError(f"Alpha Vantage: {str(payload[key])[:120]}")
        return payload

    def _fetch_quote(self, ticker: str) -> Quote:
        data = self._request({"function": "GLOBAL_QUOTE", "symbol": ticker}).get("Global Quote") or {}
        if not data.get("05. price"):
            raise MarketDataError("unknown ticker")
        return Quote(
            ticker=ticker,
            price=float(data["05. price"]),
            change=float(data.get("09. change") or 0),
            change_pct=float(str(data.get("10. change percent") or "0").rstrip("%")),
            as_of=datetime.now(UTC),
            source="alpha_vantage",
        )

    def _fetch_history(self, ticker: str) -> PriceHistory:
        series = self._request({"function": "TIME_SERIES_WEEKLY_ADJUSTED", "symbol": ticker}).get(
            "Weekly Adjusted Time Series") or {}
        if not series:
            raise MarketDataError("unknown ticker")
        cutoff = (datetime.now(UTC) - timedelta(days=365 * HISTORY_YEARS)).date().isoformat()
        points = [PricePoint(date=day, close=float(bar["5. adjusted close"]))
                  for day, bar in sorted(series.items()) if day >= cutoff]
        return PriceHistory(ticker=ticker, points=points, as_of=datetime.now(UTC), source="alpha_vantage")


def summarize_trend(history: PriceHistory) -> TrendSummary:
    """Changes over 1 month, 6 months and 1 year, moving averages and volatility from weekly closes."""
    closes = [p.close for p in history.points]
    last = closes[-1]

    def change(weeks: int) -> float | None:
        if len(closes) <= weeks or closes[-1 - weeks] == 0:
            return None
        return last / closes[-1 - weeks] - 1

    def average(weeks: int) -> float | None:
        return sum(closes[-weeks:]) / weeks if len(closes) >= weeks else None

    year = closes[-(WEEKS_PER_YEAR + 1):]
    returns = [b / a - 1 for a, b in pairwise(year) if a]
    volatility = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        volatility = math.sqrt(variance) * math.sqrt(WEEKS_PER_YEAR)

    return TrendSummary(
        ticker=history.ticker, last_close=last, last_date=history.points[-1].date,
        change_1m=change(4), change_6m=change(26), change_1y=change(WEEKS_PER_YEAR),
        sma_10w=average(10), sma_40w=average(40), volatility=volatility,
        as_of=history.as_of, source=history.source,
    )
