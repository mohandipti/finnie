"""Retrieval evaluation against the real FAISS index and embedding model.

Makes about 90 embedding calls, so it's excluded from the default run:

    pytest -m eval          # these tests
    finnie-eval-retrieval   # the same evaluation as a readable report

Skipped when the index hasn't been built or no Google API key is configured.
"""

import pytest

from src.core.config import ConfigurationError, get_settings, min_relevance_score
from src.rag.evaluate import article_index, evaluate, format_report, load_golden
from src.rag.index import read_manifest

pytestmark = pytest.mark.eval

# The Market, Portfolio and Goal Planning agents retrieve 3 chunks; Finance Q&A and Tax retrieve 5.
MIN_HIT_RATE_AT_3 = 0.95
MIN_HIT_RATE_AT_5 = 0.95
MIN_MRR = 0.85
MIN_USABLE_RATE = 0.95
MIN_OFF_TOPIC_REJECTION = 0.85


@pytest.fixture(scope="module")
def retriever():
    settings = get_settings()
    if settings.embedding_provider == "google" and settings.google_api_key is None:
        pytest.skip("GOOGLE_API_KEY is not set")
    from src.rag.faiss_retriever import FaissRetriever

    try:
        return FaissRetriever.from_settings()
    except ConfigurationError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def report(retriever):
    queries, off_topic = load_golden()
    report = evaluate(retriever, queries, off_topic, article_index(), k=5, min_score=min_relevance_score())
    print("\n" + format_report(report))  # shown with pytest -s
    return report


def test_index_is_built_from_the_current_articles():
    manifest = read_manifest(get_settings().faiss_index_dir)
    if manifest is None:
        pytest.skip("the FAISS index hasn't been built")
    assert manifest["articles"] == len(article_index()), "articles changed; rebuild with finnie-index --rebuild"


def test_the_right_article_is_retrieved(report):
    misses = [r.query.question for r in report.misses()]
    assert report.hit_rate(3) >= MIN_HIT_RATE_AT_3, misses
    assert report.hit_rate(5) >= MIN_HIT_RATE_AT_5, misses
    assert report.mrr() >= MIN_MRR


def test_hits_clear_the_agents_minimum_score(report):
    assert report.usable_rate() >= MIN_USABLE_RATE


def test_off_topic_questions_find_nothing_relevant(report):
    assert report.off_topic_rejection_rate() >= MIN_OFF_TOPIC_REJECTION, report.off_topic


@pytest.mark.parametrize(("question", "categories"), [
    ("How are dividends taxed?", ["taxes", "retirement"]),
    ("What is volatility?", ["market_concepts", "risk", "stocks", "etfs"]),
    ("How should I save for a house?", ["goal_planning", "basics", "retirement"]),
])
def test_category_filter_returns_only_those_categories(retriever, question, categories):
    docs = retriever.retrieve(question, k=5, categories=categories)
    assert docs
    assert {d.source.category for d in docs} <= set(categories)
