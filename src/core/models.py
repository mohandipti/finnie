"""Core data models shared across agents, workflow, market data and UI.

All agent inputs/outputs use these Pydantic models so that data arriving from
untrusted sources (CSV uploads, form inputs, LLM extraction, market APIs) is
validated at the boundary.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

KnowledgeLevel = Literal["beginner", "intermediate", "advanced"]
RiskTolerance = Literal["conservative", "moderate", "aggressive"]
AssetType = Literal["stock", "etf", "mutual_fund", "bond", "cash", "other"]
SourceKind = Literal["kb", "market_api", "news_api"]
QuoteSource = Literal["alpha_vantage", "cache", "mock"]
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
