from datetime import UTC, datetime, timedelta

import diskcache
import requests

from src.data.news_data import MAX_ARTICLES, NewsService, parse_articles

NOW = datetime(2026, 11, 2, 18, 0, tzinfo=UTC)


def article(n: int, days_old: float = 0) -> dict:
    return {"position": n, "title": f"Headline {n}", "link": f"https://example.com/{n}",
            "source": {"name": "Example News"}, "iso_date": (NOW - timedelta(days=days_old)).isoformat()}


AAPL_NEWS = {"news_results": [article(1), article(2)]}
NO_RESULTS = {"error": "Google hasn't returned any results for this query."}
QUOTA_USED = {"error": "Your account has run out of searches."}


class FakeResponse:
    def __init__(self, payload, ok=True):
        self.payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 429

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params["q"])
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload, ok="error" not in payload or "hasn't returned" in payload["error"])


def service(tmp_path, session, api_key="key", ttl=3600) -> NewsService:
    return NewsService(api_key=api_key, cache=diskcache.Cache(tmp_path), ttl_seconds=ttl, session=session)


def test_search_is_parsed_and_then_served_from_cache(tmp_path):
    session = FakeSession(AAPL_NEWS)
    news = service(tmp_path, session)

    live = news.search("AAPL  stock")
    assert (live.query, live.source, len(live.articles)) == ("AAPL stock", "serpapi", 2)
    assert live.articles[0].publisher == "Example News" and live.articles[0].url == "https://example.com/1"

    cached = news.search("aapl stock")
    assert cached.source == "cache" and cached.articles == live.articles
    assert session.calls == ["AAPL stock"]


def test_failure_falls_back_to_the_last_good_result(tmp_path):
    session = FakeSession(AAPL_NEWS, QUOTA_USED, requests.ConnectionError())
    news = service(tmp_path, session, ttl=0)  # expires at once, so every search goes to the API
    news.search("AAPL stock")

    assert news.search("AAPL stock").source == "cache"
    assert news.search("AAPL stock").articles[0].title == "Headline 1"
    assert session.calls == ["AAPL stock", "AAPL stock", "AAPL stock"]


def test_no_results_is_an_empty_result_not_an_error(tmp_path):
    result = service(tmp_path, FakeSession(NO_RESULTS)).search("zzzz qqqq")
    assert result.articles == [] and result.source == "serpapi"


def test_without_a_key_only_cached_results_are_used(tmp_path):
    session = FakeSession()
    assert service(tmp_path, session, api_key=None).search("AAPL stock") is None
    assert session.calls == []
    assert service(tmp_path, session).search("   ") is None


def test_parse_drops_old_and_incomplete_items_and_unwraps_story_clusters():
    results = [
        article(1, days_old=45),
        {"position": 2, "title": "No link"},
        {"position": 3, "highlight": article(3), "stories": [article(4)]},
        *[article(n) for n in range(5, 15)],
    ]
    articles = parse_articles(results, NOW)
    assert len(articles) == MAX_ARTICLES
    assert [a.title for a in articles[:2]] == ["Headline 3", "Headline 5"]
