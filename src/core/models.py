"""Core data models shared across agents, workflow, market data and UI.

All agent inputs/outputs use these Pydantic models so that data arriving from
untrusted sources (CSV uploads, form inputs, LLM extraction, market APIs) is
validated at the boundary.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

KnowledgeLevel = Literal["beginner", "intermediate", "advanced"]
RiskTolerance = Literal["conservative", "moderate", "aggressive"]
AssetType = Literal["stock", "etf", "mutual_fund", "bond", "cash", "other"]
SourceKind = Literal["kb", "market_api", "news_api"]
QuoteSource = Literal["alpha_vantage", "cache", "mock"]
NewsSource = Literal["serpapi", "cache"]
Market = Literal["US"]  # only the US stock market is supported for now
AgentName = Literal["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]


def _normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")
    return ticker


class FinancialGoal(BaseModel):
    """A savings goal used by the Goal Planning Agent."""

    name: str = Field(min_length=1)  # e.g. "Retirement", "House down payment"
    target_amount: float = Field(gt=0)
    current_savings: float = Field(default=0, ge=0)
    monthly_contribution: float = Field(default=0, ge=0)
    horizon_years: float = Field(gt=0)  # years from now until the goal is due
    risk_tolerance: RiskTolerance
    # Dollars of the user's portfolio set aside for this goal; None means not decided yet, so Finnie asks.
    from_portfolio: float | None = Field(default=None, ge=0)


class UserProfile(BaseModel):
    """Per-session user profile used to personalize responses."""

    session_id: str
    knowledge_level: KnowledgeLevel = "beginner"
    risk_tolerance: RiskTolerance | None = None
    goals: list[FinancialGoal] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)  # e.g. ["ETFs", "retirement"], inferred from conversation


class Holding(BaseModel):
    """A single position in a user's portfolio."""

    ticker: str
    quantity: float = Field(gt=0)
    cost_basis: float | None = Field(default=None, ge=0)  # per unit
    asset_type: AssetType | None = None  # auto-detected if None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return _normalize_ticker(value)


class Portfolio(BaseModel):
    """A user's holdings, analyzed by the Portfolio Analysis Agent."""

    holdings: list[Holding] = Field(min_length=1)
    market: Market = "US"


class Source(BaseModel):
    """A citation attached to a response."""

    title: str
    url: str | None = None
    category: str | None = None
    snippet: str | None = None
    kind: SourceKind


class RetrievedDocument(BaseModel):
    """A knowledge base chunk returned by the retriever."""

    content: str
    score: float = Field(ge=0, le=1)  # normalized similarity; higher is more relevant
    source: Source


class AgentResponse(BaseModel):
    """Standard output of every agent's run()."""

    agent: str
    content: str  # markdown
    sources: list[Source] = Field(default_factory=list)
    data: dict[str, Any] | None = None  # structured payload for UI charts (metrics, projections, quotes)
    confidence: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None


class ExtractedEntities(BaseModel):
    """Entities the router pulls out of the user's message for downstream agents."""

    tickers: list[str] = Field(default_factory=list)
    amounts: list[float] = Field(default_factory=list)  # dollar amounts mentioned
    horizon_years: float | None = None
    holdings: list[Holding] = Field(default_factory=list)  # e.g. "I have 10 AAPL and 5 VTI"

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(_normalize_ticker(t) for t in value if t.strip()))


class RouterDecision(BaseModel):
    """Structured output of the workflow router."""

    agents: list[AgentName] = Field(min_length=1, description="Every agent needed to answer the message")
    reason: str = Field(description="One short sentence explaining the choice")
    extracted: ExtractedEntities = Field(default_factory=ExtractedEntities)
    scope: Literal["finance", "off_topic", "disallowed"] = Field(
        default="finance",
        description="finance for anything about money; off_topic for unrelated requests; disallowed for requests "
                    "to help break the law or harm someone",
    )


class Quote(BaseModel):
    """A point-in-time price quote with freshness metadata."""

    ticker: str
    price: float = Field(ge=0)
    change: float
    change_pct: float
    currency: str = "USD"
    as_of: datetime
    source: QuoteSource

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return _normalize_ticker(value)


class PricePoint(BaseModel):
    date: date
    close: float = Field(ge=0)  # adjusted for splits and dividends


class PriceHistory(BaseModel):
    """Weekly closing prices, oldest first, with freshness metadata."""

    ticker: str
    interval: Literal["weekly"] = "weekly"
    points: list[PricePoint] = Field(min_length=1)
    as_of: datetime  # when the data was fetched
    source: QuoteSource


class TrendSummary(BaseModel):
    """Simple trend statistics computed from a price history, for the Market agent and tab."""

    ticker: str
    last_close: float
    last_date: date
    change_1m: float | None  # fractional change, e.g. 0.05 = +5%; None if the history is too short
    change_6m: float | None
    change_1y: float | None
    sma_10w: float | None  # ≈ 50-day moving average
    sma_40w: float | None  # ≈ 200-day moving average
    volatility: float | None  # annualized, from weekly returns over the last year
    as_of: datetime
    source: QuoteSource


class NewsArticle(BaseModel):
    """One headline from a news search."""

    title: str
    url: str
    publisher: str | None = None
    published_at: datetime | None = None


class NewsResult(BaseModel):
    """Headlines for one search query, most relevant first, with freshness metadata."""

    query: str
    articles: list[NewsArticle] = Field(default_factory=list)  # empty when nothing recent was found
    as_of: datetime  # when the search was run
    source: NewsSource
