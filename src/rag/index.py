"""Build and save the FAISS index of the knowledge base.

    python -m src.rag.index --rebuild

Chunks are embedded with the configured embedding model and stored with cosine
similarity (vectors are L2-normalized, then compared by inner product). A
manifest next to the index records which embedding model built it, so the
retriever can refuse to query it with a different model.
"""

import argparse
import json
import logging
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from src.core.config import embedding_signature, get_embeddings, get_settings
from src.rag.chunker import chunk_documents
from src.rag.loader import load_articles

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"

# Cosine similarity. FAISS.load_local() does not persist these, so the retriever
# must pass the same settings when loading or queries won't be normalized.
STORE_SETTINGS = {"normalize_L2": True, "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT}

# LangChain warns that normalization doesn't apply to inner product, but it still normalizes
# both stored and query vectors, which is what turns inner product into cosine similarity.
warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable")


def build_index(
    articles_dir: Path,
    index_dir: Path,
    embeddings: Embeddings,
    signature: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> dict:
    """Load, chunk, embed and save the knowledge base. Returns the manifest."""
    started = time.perf_counter()
    chunks = chunk_documents(load_articles(articles_dir), chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, embeddings, **STORE_SETTINGS)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))

    manifest = {
        "embedding": signature,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "articles": len({c.metadata["file"] for c in chunks}),
        "chunks": len(chunks),
        "categories": dict(Counter(c.metadata["category"] for c in chunks)),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    logger.info("built index: %d chunks in %.1fs -> %s", len(chunks), time.perf_counter() - started, index_dir)
    return manifest


def read_manifest(index_dir: Path) -> dict | None:
    path = index_dir / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    from src.utils.logger import setup_logger

    parser = argparse.ArgumentParser(description="Build the Finnie knowledge base FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if an index already exists")
    args = parser.parse_args()

    setup_logger()
    settings = get_settings()
    existing = read_manifest(settings.faiss_index_dir)
    if existing and not args.rebuild:
        print(f"Index already exists at {settings.faiss_index_dir} ({existing['chunks']} chunks). Use --rebuild.")
        return

    manifest = build_index(
        settings.knowledge_base_dir,
        settings.faiss_index_dir,
        get_embeddings(),
        embedding_signature(settings),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
