from datetime import UTC, datetime

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents.news_agent import NO_TOPIC_NOTE, NewsAgent, format_news
from src.core.models import ExtractedEntities, NewsArticle, NewsResult
from src.workflow.router import keyword_route

NOW = datetime(2026, 11, 2, 18, 0, tzinfo=UTC)
AAPL_NEWS = NewsResult(query="AAPL stock", as_of=NOW, source="serpapi", articles=[
    NewsArticle(title="Apple reports earnings", url="https://example.com/1", publisher="Quartz", published_at=NOW),
    NewsArticle(title="Apple buys an AI startup", url="https://example.com/2"),
])
FED_NEWS = NewsResult(query="Federal Reserve", as_of=NOW, source="cache", articles=[
    NewsArticle(title="Fed holds rates steady", url="https://example.com/3", publisher="Reuters", published_at=NOW),
])


class FakeNews:
    def __init__(self):
        self.searches = []
        self.results = {"AAPL stock": AAPL_NEWS, "Federal Reserve": FED_NEWS}

    def search(self, query):
        self.searches.append(query)
        return self.results.get(query)

    def cached(self, query):
        return self.results.get(query)


class FakeLLM(GenericFakeChatModel):
    """Replies with the given messages in order; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def agent(*replies: AIMessage) -> tuple[NewsAgent, FakeNews]:
    news = FakeNews()
    return NewsAgent(llm=FakeLLM(messages=iter(replies)), news=news), news


def state(query: str, tickers: list[str]) -> dict:
    return {"session_id": "s1", "user_query": query, "messages": [],
            "extracted_entities": ExtractedEntities(tickers=tickers)}


def test_prefetches_news_for_extracted_tickers_and_lists_articles_as_sources():
    news_agent, news = agent(AIMessage(content="Apple reported earnings."))
    response = news_agent.run(state("Any news on Apple?", ["AAPL", "NOPE", "MSFT"]))

    assert response.error is None and response.content == "Apple reported earnings."
    assert news.searches == ["AAPL stock", "NOPE stock"]  # capped at 2 searches
    assert [s.url for s in response.sources] == ["https://example.com/1", "https://example.com/2"]
    assert response.sources[0].kind == "news_api" and response.sources[0].category == "Quartz, Nov 02, 2026"
    assert response.sources[1].category is None


def test_without_a_ticker_the_llm_searches_a_topic():
    tool_call = AIMessage(content="", tool_calls=[{"name": "get_news", "args": {"query": "Federal Reserve"},
                                                   "id": "1"}])
    news_agent, news = agent(tool_call, AIMessage(content="The Fed held rates steady."))
    response = news_agent.run(state("What's the latest on interest rates?", []))

    assert response.content == "The Fed held rates steady."
    assert news.searches == ["Federal Reserve"]
    assert "Fed holds rates steady (Reuters" in response.data["tool_results"][0]["result"]
    assert [s.title for s in response.sources] == ["Fed holds rates steady"]


def test_prompt_includes_headlines_or_a_no_topic_note():
    news_agent, _ = agent()
    s = state("Any news on AAPL?", ["AAPL"])
    message = news_agent.build_user_message(s, [], news_agent.prefetch_data(s))
    assert "News headlines:" in message and "Today's date:" in message
    assert message.endswith("Question: Any news on AAPL?")

    assert NO_TOPIC_NOTE in news_agent.build_user_message(state("Any news?", []), [], None)


def test_format_news():
    text = format_news("AAPL stock", AAPL_NEWS)
    assert '"AAPL stock" [live from Google News via SerpAPI, retrieved 2026-11-02 18:00 UTC]' in text
    assert "- Apple reports earnings (Quartz, 2026-11-02 18:00 UTC)" in text
    assert "- Apple buys an AI startup (unknown publisher, date unknown)" in text
    assert "cached" in format_news("Federal Reserve", FED_NEWS)
    assert "unavailable" in format_news("NOPE stock", None)
    empty = NewsResult(query="zzzz", as_of=NOW, source="serpapi")
    assert "No articles from the last month" in format_news("zzzz", empty)


def test_keyword_fallback_routes_news_questions():
    assert keyword_route("Any news on AAPL?").agents == ["news"]
    assert keyword_route("Latest headlines about inflation").agents == ["news"]
