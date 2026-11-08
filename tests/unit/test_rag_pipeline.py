"""Offline tests for the RAG pipeline: article loading, chunking, the knowledge base itself,
the golden set and the evaluation metrics. The retrieval evaluation against the real index is
in tests/integration/test_retrieval_eval.py."""

from urllib.parse import urlparse

import pytest
from langchain_core.documents import Document

from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.tax_agent import TaxAgent
from src.core.models import RetrievedDocument, Source
from src.rag.chunker import chunk_documents
from src.rag.evaluate import ARTICLES_DIR, GoldenQuery, article_index, evaluate, first_match, load_golden
from src.rag.loader import ArticleError, load_article, load_articles

CHUNK_SIZE = 1500

ARTICLE = """---
title: "What Is a Bond?"
category: bonds
source_name: Investor.gov
source_url: https://www.investor.gov/bonds
---

# What Is a Bond?

## How bonds work

A bond is a loan to a government or company.

## Risks

Interest rate risk and credit risk.
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def articles():
    return load_articles(ARTICLES_DIR)


@pytest.fixture(scope="module")
def chunks(articles):
    return chunk_documents(articles)


# Loader

def test_article_front_matter_becomes_metadata(tmp_path):
    doc = load_article(write(tmp_path, "what-is-a-bond.md", ARTICLE))
    assert doc.metadata == {"title": "What Is a Bond?", "category": "bonds", "source_name": "Investor.gov",
                            "source_url": "https://www.investor.gov/bonds", "file": "what-is-a-bond.md"}
    assert doc.page_content.startswith("# What Is a Bond?")


def test_article_missing_a_source_is_rejected(tmp_path):
    with pytest.raises(ArticleError, match="source_url"):
        load_article(write(tmp_path, "bad.md", ARTICLE.replace("source_url: https://www.investor.gov/bonds\n", "")))


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ArticleError, match="no Markdown articles"):
        load_articles(tmp_path)


# Chunker

def test_chunks_follow_sections_and_carry_their_context(tmp_path):
    chunks = chunk_documents([load_article(write(tmp_path, "what-is-a-bond.md", ARTICLE))])
    assert [c.metadata["section"] for c in chunks] == ["How bonds work", "Risks"]
    assert chunks[0].page_content.startswith("What Is a Bond? > How bonds work\n\n")
    assert [c.metadata["chunk_id"] for c in chunks] == ["what-is-a-bond.md#0", "what-is-a-bond.md#1"]
    assert chunks[1].metadata["source_url"] == "https://www.investor.gov/bonds"


def test_long_section_is_split_with_overlap():
    sentences = " ".join(f"Sentence number {i} about diversification." for i in range(200))
    doc = Document(page_content=f"## Long\n\n{sentences}", metadata={"title": "T", "file": "t.md"})
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=100)
    bodies = [c.page_content.split("\n\n", 1)[1] for c in chunks]
    assert len(chunks) > 1
    assert all(len(b) <= 500 for b in bodies)
    assert bodies[0][-40:] in bodies[1]  # the overlap repeats the end of the previous chunk


# The knowledge base itself

def test_every_article_loads_with_a_web_source(articles):
    assert len(articles) >= 80
    for doc in articles:
        url = urlparse(doc.metadata["source_url"])
        assert url.scheme == "https" and url.netloc, doc.metadata["file"]


def test_article_titles_are_unique(articles):
    # Retrieved chunks are matched back to articles by title.
    titles = [d.metadata["title"] for d in articles]
    assert len(titles) == len(set(titles))


def test_every_article_produces_chunks_within_the_size_limit(articles, chunks):
    assert {c.metadata["file"] for c in chunks} == {d.metadata["file"] for d in articles}
    for chunk in chunks:
        body = chunk.page_content.split("\n\n", 1)[1]
        assert len(body) <= CHUNK_SIZE, chunk.metadata["chunk_id"]


def test_chunk_ids_are_unique(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_agent_retrieval_categories_exist_in_the_knowledge_base(articles):
    # A misspelt category would silently filter out every chunk.
    categories = {d.metadata["category"] for d in articles}
    for agent in (TaxAgent, GoalPlanningAgent, MarketAgent, PortfolioAgent):
        assert set(agent.retrieval_categories) <= categories, agent.__name__


# Golden set

def test_golden_set_covers_every_article_and_names_only_real_ones():
    queries, off_topic = load_golden()
    expected = {f for q in queries for f in q.expected}
    assert expected == set(article_index())
    assert len({q.question for q in queries}) == len(queries)
    assert off_topic


# Metrics

def result(title, score):
    return RetrievedDocument(content="", score=score, source=Source(title=title, kind="kb"))


class FakeRetriever:
    def __init__(self, answers):
        self.answers = answers

    def retrieve(self, query, k=5, categories=None):
        return self.answers[query][:k]


def test_first_match_is_one_based():
    assert first_match(["A", "B", "C"], {"C", "B"}) == 2
    assert first_match(["A"], {"Z"}) is None


def test_report_metrics():
    articles = {"a.md": ("A", "basics"), "b.md": ("B", "taxes"), "c.md": ("C", "taxes")}
    retriever = FakeRetriever({
        "qa": [result("A", 0.80), result("B", 0.70)],   # rank 1
        "qb": [result("A", 0.70), result("B", 0.60)],   # rank 2, but below the min score
        "qc": [result("A", 0.70), result("B", 0.65)],   # not found
        "pizza": [result("A", 0.55)],                     # rejected
        "weather": [result("B", 0.63)],                   # accepted: would wrongly get context
    })
    queries = [GoldenQuery("qa", ("a.md",)), GoldenQuery("qb", ("b.md",)), GoldenQuery("qc", ("c.md",))]
    report = evaluate(retriever, queries, ["pizza", "weather"], articles, k=5, min_score=0.62)

    assert report.hit_rate(1) == pytest.approx(1 / 3)
    assert report.hit_rate(5) == pytest.approx(2 / 3)
    assert report.mrr() == pytest.approx((1 + 1 / 2) / 3)
    assert report.usable_rate() == pytest.approx(1 / 3)
    assert report.off_topic_rejection_rate() == pytest.approx(1 / 2)
    assert [r.query.question for r in report.misses()] == ["qc"]
    assert report.by_category() == {"basics": (1, 1), "taxes": (1, 2)}
