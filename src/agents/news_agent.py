"""News Synthesizer Agent: recent headlines about a company or topic, explained for learners.

Headlines for the tickers the router extracted are pre-fetched in code (searched as
"<TICKER> stock"), so "Any news on AAPL?" needs a single LLM call. For a topic such as
interest rates or inflation, the LLM writes the search itself with the `get_news` tool.
Every article shown to the LLM is listed as a source, linking to the original story.
"""

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from src.agents.base_agent import BaseAgent
from src.agents.prompts import NEWS_SYSTEM_PROMPT
from src.core.models import AgentResponse, NewsResult, RetrievedDocument, Source
from src.data.market_data import TICKER_PATTERN
from src.data.news_data import NewsService
from src.workflow.state import FinnieState

MAX_QUERIES = 2  # each search uses one of the free plan's 250 a month

SOURCE_LABELS = {"serpapi": "live from Google News via SerpAPI", "cache": "cached from Google News via SerpAPI"}

NO_TOPIC_NOTE = (
    "No ticker was identified in the message. Call get_news with a short search query for the "
    "company or topic the user asks about (for example \"NVDA stock\" or \"Federal Reserve interest "
    "rates\"); for general market news use \"stock market\". If it's unclear what they mean, ask."
)


def format_news(query: str, result: NewsResult | None) -> str:
    """A plain-text block of headlines for the prompt."""
    if result is None:
        return f'News for "{query}": unavailable (the news provider can\'t be reached right now).'
    retrieved = f"{SOURCE_LABELS[result.source]}, retrieved {result.as_of:%Y-%m-%d %H:%M} UTC"
    lines = [f'News for "{result.query}" [{retrieved}]']
    if not result.articles:
        lines.append("- No articles from the last month were found.")
    for article in result.articles:
        when = f"{article.published_at:%Y-%m-%d %H:%M} UTC" if article.published_at else "date unknown"
        lines.append(f"- {article.title} ({article.publisher or 'unknown publisher'}, {when})")
    return "\n".join(lines)


def news_sources(results: list[NewsResult]) -> list[Source]:
    """One Source per distinct article, in the order the LLM saw them."""
    seen: set[str] = set()
    sources = []
    for result in results:
        for article in result.articles:
            if article.url in seen:
                continue
            seen.add(article.url)
            details = [article.publisher, f"{article.published_at:%b %d, %Y}" if article.published_at else None]
            sources.append(Source(title=article.title, url=article.url, kind="news_api",
                                  category=", ".join(d for d in details if d) or None))
    return sources


class NewsAgent(BaseAgent):
    name = "news"
    description = "Recent financial news about a company, ticker or topic"
    system_prompt = NEWS_SYSTEM_PROMPT

    def __init__(self, news: NewsService | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.news = news or NewsService.from_settings()

    def queries(self, state: FinnieState) -> list[str]:
        entities = state.get("extracted_entities")
        tickers = [t for t in (entities.tickers if entities else []) if TICKER_PATTERN.match(t)]
        return [f"{t} stock" for t in tickers[:MAX_QUERIES]]

    def prefetch_data(self, state: FinnieState) -> dict[str, Any] | None:
        queries = self.queries(state)
        if not queries:
            return None
        results = {q: self.news.search(q) for q in queries}
        return {
            "queries": queries,
            "news": {q: r.model_dump(mode="json") for q, r in results.items() if r},
            "summary": "\n\n".join(format_news(q, r) for q, r in results.items()),
        }

    def get_tools(self) -> list[BaseTool]:
        def get_news(query: str) -> str:
            """Recent news headlines for a company, ticker or financial topic, e.g. "NVDA stock" or "inflation"."""
            return format_news(query, self.news.search(query))

        return [StructuredTool.from_function(get_news)]

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        parts = [f"News headlines:\n\n{data['summary']}" if data else NO_TOPIC_NOTE,
                 f"Today's date: {datetime.now(UTC):%Y-%m-%d}",
                 f"Question: {state.get('user_query', '')}"]
        return "\n\n".join(parts)

    def build_response(
        self, state: FinnieState, content: str, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> AgentResponse:
        data = data or {}
        results = [NewsResult.model_validate(r) for r in data.get("news", {}).values()]
        # Searches the LLM ran with get_news were just cached, so reading them back costs no request.
        for call in data.get("tool_results", []):
            if call["tool"] == "get_news" and (result := self.news.cached(call["args"].get("query", ""))):
                results.append(result)
        return AgentResponse(agent=self.name, content=content, sources=news_sources(results), data=data or None)
