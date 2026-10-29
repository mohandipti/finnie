"""Retriever interface used by agents that ground answers in the knowledge base.

Agents depend on this protocol rather than on FAISS directly, so they can be
tested with an in-memory fake and run (without citations) before the index exists.
"""

from typing import Protocol

from src.core.models import RetrievedDocument


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5, categories: list[str] | None = None) -> list[RetrievedDocument]:
        """Return up to `k` chunks most relevant to `query`, best first.

        When `categories` is given, only chunks whose source category is in it are returned.
        """
        ...
