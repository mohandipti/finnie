"""`Retriever` implementation backed by the FAISS index built by `src.rag.index`."""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.core.config import ConfigurationError, embedding_signature, get_embeddings, get_settings
from src.core.models import RetrievedDocument, Source
from src.rag.index import STORE_SETTINGS, read_manifest

logger = logging.getLogger(__name__)


class FaissRetriever:
    def __init__(self, index_dir: Path, embeddings: Embeddings, signature: str):
        manifest = read_manifest(index_dir)
        if manifest is None:
            raise ConfigurationError(f"No FAISS index at {index_dir}. Build it with: python -m src.rag.index --rebuild")
        if manifest["embedding"] != signature:
            raise ConfigurationError(
                f"FAISS index was built with {manifest['embedding']} but the configured embedding model is "
                f"{signature}. Rebuild it with: python -m src.rag.index --rebuild"
            )
        # The index is built locally by src.rag.index, so loading its pickled docstore is safe.
        self.store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True, **STORE_SETTINGS
        )
        logger.info("loaded FAISS index: %d chunks from %s", manifest["chunks"], index_dir)

    @classmethod
    def from_settings(cls) -> "FaissRetriever":
        settings = get_settings()
        return cls(settings.faiss_index_dir, get_embeddings(), embedding_signature(settings))

    def retrieve(self, query: str, k: int = 5, categories: list[str] | None = None) -> list[RetrievedDocument]:
        """Top-k chunks by cosine similarity, optionally limited to the given categories."""
        allowed = set(categories) if categories else None
        results = self.store.similarity_search_with_score(
            query,
            k=k,
            filter=(lambda metadata: metadata.get("category") in allowed) if allowed else None,
            fetch_k=max(20, k * 4),  # search more candidates when filtering so k results survive
        )
        return [
            RetrievedDocument(
                content=doc.page_content,
                score=max(0.0, min(1.0, float(similarity))),  # cosine similarity, clipped to [0, 1]
                source=Source(
                    title=doc.metadata["title"],
                    url=doc.metadata.get("source_url"),
                    category=doc.metadata.get("category"),
                    snippet=doc.page_content[:200],
                    kind="kb",
                ),
            )
            for doc, similarity in results
        ]
