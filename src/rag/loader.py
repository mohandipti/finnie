"""Load knowledge base articles (Markdown with YAML front-matter) as LangChain Documents."""

import logging
from pathlib import Path

import frontmatter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("title", "category", "source_name", "source_url")


class ArticleError(ValueError):
    """Raised when an article is missing required front-matter."""


def load_article(path: Path) -> Document:
    post = frontmatter.load(path)
    missing = [field for field in REQUIRED_FIELDS if not post.metadata.get(field)]
    if missing:
        raise ArticleError(f"{path.name}: missing front-matter field(s) {missing}")
    metadata = {
        "title": str(post.metadata["title"]),
        "category": str(post.metadata["category"]),
        "source_name": str(post.metadata["source_name"]),
        "source_url": str(post.metadata["source_url"]),
        "level": str(post.metadata.get("level", "beginner")),
        "file": path.name,
    }
    return Document(page_content=post.content.strip(), metadata=metadata)


def load_articles(directory: Path) -> list[Document]:
    """Load every `*.md` article in `directory`, sorted by file name."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ArticleError(f"no Markdown articles found in {directory}")
    documents = [load_article(path) for path in paths]
    logger.info("loaded %d articles from %s", len(documents), directory)
    return documents
