"""Financial news headlines from Google News via SerpAPI, cached on disk.

SerpAPI's free plan allows 250 searches a month, so:

- a search is reused for NEWS_CACHE_MINUTES (60 by default) before it is run again;
- the last good result of each search is kept indefinitely, and returned (marked "cache")
  when a search fails or the monthly quota is used up.

Google News results have a headline, publisher, link and publish time, but no summary,
so agents work from headlines only.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import diskcache
import requests

from src.core.config import Settings, get_settings
from src.core.models import NewsArticle, NewsResult

logger = logging.getLogger(__name__)

API_URL = "https://serpapi.com/search.json"
MAX_ARTICLES = 6
MAX_AGE_DAYS = 30  # older articles are dropped, so "latest news" isn't answered with old stories
NO_RESULTS = "hasn't returned any results"  # SerpAPI's error text for an empty search


class NewsDataError(RuntimeError):
    """A search couldn't be run (network error, invalid key or quota used up)."""


def normalize_query(query: str) -> str:
    return " ".join(query.split())


def parse_articles(results: list[dict[str, Any]], now: datetime) -> list[NewsArticle]:
    """The most relevant recent articles; a story cluster is represented by its highlighted article."""
    cutoff = now - timedelta(days=MAX_AGE_DAYS)
    articles = []
    for item in results:
        if not item.get("title"):
            item = item.get("highlight") or {}
        if not item.get("title") or not item.get("link"):
            continue
        try:
            published = datetime.fromisoformat(item["iso_date"]) if item.get("iso_date") else None
        except ValueError:
            published = None
        if published is not None and published < cutoff:
            continue
        articles.append(NewsArticle(title=item["title"], url=item["link"],
                                    publisher=(item.get("source") or {}).get("name"), published_at=published))
        if len(articles) == MAX_ARTICLES:
            break
    return articles


class NewsService:
    def __init__(self, api_key: str | None, cache: diskcache.Cache, ttl_seconds: int = 3600,
                 session: requests.Session | None = None, timeout: float = 20) -> None:
        self.api_key = api_key
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.session = session or requests.Session()
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "NewsService":
        settings = settings or get_settings()
        key = settings.serp_api_key.get_secret_value() if settings.serp_api_key else None
        return cls(api_key=key or None, cache=diskcache.Cache(settings.market_cache_dir),
                   ttl_seconds=settings.news_cache_minutes * 60)

    def search(self, query: str) -> NewsResult | None:
        """Recent headlines for a query, or None if there is neither a live nor a cached result."""
        query = normalize_query(query)
        if not query:
            return None
        fresh = self.cache.get(f"news:{query.lower()}")
        if fresh is not None:
            return NewsResult.model_validate(fresh).model_copy(update={"source": "cache"})
        if self.api_key:
            try:
                result = self._fetch(query)
            except NewsDataError as exc:
                logger.warning("news for %r unavailable: %s", query, exc)
            else:
                data = result.model_dump(mode="json")
                self.cache.set(f"news:{query.lower()}", data, expire=self.ttl_seconds)
                self.cache.set(f"last-news:{query.lower()}", data)
                return result
        return self.cached(query)

    def cached(self, query: str) -> NewsResult | None:
        """The last good result for a query, without searching."""
        last = self.cache.get(f"last-news:{normalize_query(query).lower()}")
        return NewsResult.model_validate(last).model_copy(update={"source": "cache"}) if last else None

    def _fetch(self, query: str) -> NewsResult:
        params = {"engine": "google_news", "q": query, "gl": "us", "hl": "en", "api_key": self.api_key}
        try:
            response = self.session.get(API_URL, params=params, timeout=self.timeout)
            payload: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise NewsDataError(type(exc).__name__) from exc
        now = datetime.now(UTC)
        error = payload.get("error")
        if error and NO_RESULTS in error:
            return NewsResult(query=query, as_of=now, source="serpapi")
        if error or not response.ok:
            # Never include the request URL in the message: it contains the API key.
            raise NewsDataError(f"SerpAPI: {str(error or response.status_code)[:120]}")
        return NewsResult(query=query, articles=parse_articles(payload.get("news_results") or [], now),
                          as_of=now, source="serpapi")
