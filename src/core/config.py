"""Application settings and model factories.

Settings are read from environment variables, or from a `.env` file in the
project root (see `.env.example`). API keys are kept as `SecretStr` so they
never appear in logs or reprs.

Only GOOGLE_API_KEY is required: it is used by both the Gemini chat model and
Google embeddings. FAISS runs locally and needs no key, only an index location
and the embedding model that built the index.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.models import AgentName

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Components that call an LLM and can each be given their own model settings.
LLM_COMPONENTS: tuple[str, ...] = ("router", "synthesizer", *get_args(AgentName))


class ConfigurationError(RuntimeError):
    """Raised when a required setting, such as an API key, is missing."""


class LLMOverride(BaseModel):
    """Per-component LLM settings; unset fields fall back to the global defaults."""

    model: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    thinking_budget: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    google_api_key: SecretStr | None = None  # Gemini chat model and Google embeddings
    alpha_vantage_api_key: SecretStr | None = None  # live market data;  
    serp_api_key: SecretStr | None = None  # live news feed; 

    # LLM defaults, used by every component unless overridden in AGENT_LLM
    gemini_model: str = "gemini-3.5-flash"  # gemini-2.0-flash (the spec default) is retired
    llm_temperature: float = 0.0 #Make the answer consistent everytime you call it
    llm_timeout_seconds: float = 30
    # Tokens Gemini may spend "thinking" before answering; most of the latency on Gemini 3.x Flash.
    # 0 disables it: measured ~3.5s per turn instead of ~9s, with equally good answers and routing.
    # None keeps the model's default.
    llm_thinking_budget: int | None = 0
    # Per-component overrides as JSON, e.g.
    # AGENT_LLM='{"portfolio": {"model": "gemini-2.5-pro"}, "finance_qa": {"temperature": 0.3}}'
    agent_llm: dict[str, LLMOverride] = {}

    # Embeddings and FAISS. Index and query must use the same embedding model;
    # changing it requires rebuilding the index.
    embedding_provider: Literal["google", "local"] = "google"
    google_embedding_model: str = "models/gemini-embedding-001"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_dir: Path = PROJECT_ROOT / "data" / "index"

    # Knowledge base and chunking
    knowledge_base_dir: Path = PROJECT_ROOT / "src" / "data" / "knowledge_base" / "articles"
    chunk_size: int = 1500  # characters (~350 tokens); articles are split by section first
    chunk_overlap: int = 150

    # Retrieval. Chunks scoring below the minimum cosine similarity are discarded.
    # None uses a default calibrated for the embedding provider (see MIN_SCORE_DEFAULTS).
    retrieval_min_score: float | None = None

    log_level: str = "INFO"

    @field_validator("agent_llm")
    @classmethod
    def check_components(cls, value: dict[str, LLMOverride]) -> dict[str, LLMOverride]:
        unknown = sorted(set(value) - set(LLM_COMPONENTS))
        if unknown:
            raise ValueError(f"unknown component(s) {unknown}; expected any of {list(LLM_COMPONENTS)}")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _require_google_key(settings: Settings) -> SecretStr:
    if settings.google_api_key is None or not settings.google_api_key.get_secret_value():
        raise ConfigurationError(
            "GOOGLE_API_KEY is not set. Add it to your environment or to .env (see .env.example)."
        )
    return settings.google_api_key


@lru_cache
def get_llm(component: str) -> BaseChatModel:
    """The Gemini chat model for one component (an agent name, "router" or "synthesizer").

    Each component gets its own instance, built from the global defaults plus its
    AGENT_LLM override. Repeated calls for the same component return the same instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if component not in LLM_COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {list(LLM_COMPONENTS)}")
    settings = get_settings()
    override = settings.agent_llm.get(component, LLMOverride())
    thinking = settings.llm_thinking_budget if override.thinking_budget is None else override.thinking_budget
    return ChatGoogleGenerativeAI(
        model=override.model or settings.gemini_model,
        google_api_key=_require_google_key(settings),
        temperature=settings.llm_temperature if override.temperature is None else override.temperature,
        timeout=override.timeout_seconds or settings.llm_timeout_seconds,
        max_retries=0,  # retries are handled by with_retry() in the router and BaseAgent
        **({"thinking_budget": thinking} if thinking is not None else {}),
    )


# Embedding models differ in baseline similarity. With gemini-embedding-001, on-topic
# questions score ~0.66-0.79 against the knowledge base and off-topic ones ~0.50-0.55.
# The local value is uncalibrated; check it with real queries before relying on it.
MIN_SCORE_DEFAULTS = {"google": 0.62, "local": 0.35}


def min_relevance_score(settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    if settings.retrieval_min_score is not None:
        return settings.retrieval_min_score
    return MIN_SCORE_DEFAULTS[settings.embedding_provider]


def embedding_signature(settings: Settings | None = None) -> str:
    """Identifies the configured embedding model; stored with the FAISS index to detect mismatches."""
    settings = settings or get_settings()
    if settings.embedding_provider == "local":
        return f"local:{settings.local_embedding_model}"
    return f"google:{settings.google_embedding_model}"


@lru_cache
def get_embeddings() -> Embeddings:
    """The embedding model used to build and query the FAISS index."""
    settings = get_settings()
    if settings.embedding_provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.google_embedding_model,
        google_api_key=_require_google_key(settings),
    )
