"""Retrieval evaluation: how well the FAISS index finds the right article for a question.

    finnie-eval-retrieval            # report on the golden set
    finnie-eval-retrieval -k 3       # the cut-off used by the Market, Portfolio and Goal Planning agents

The golden set (src/data/eval/retrieval_golden.yaml) pairs questions with the article(s) that
answer them, plus off-topic questions that should find nothing. Metrics are per question:

    hit@k   an expected article is among the top k chunks
    MRR     mean of 1 / rank of the first chunk from an expected article (0 if not in the top k)
    usable  the hit also clears the agents' minimum relevance score, so an agent would keep it

Each question costs one embedding call.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.config import PROJECT_ROOT, get_settings, min_relevance_score
from src.rag.loader import load_articles
from src.rag.retriever import Retriever

GOLDEN_SET = PROJECT_ROOT / "src" / "data" / "eval" / "retrieval_golden.yaml"
ARTICLES_DIR = PROJECT_ROOT / "src" / "data" / "knowledge_base" / "articles"


@dataclass(frozen=True)
class GoldenQuery:
    question: str
    expected: tuple[str, ...]  # article file names, any of which answers the question


@dataclass(frozen=True)
class QueryResult:
    query: GoldenQuery
    category: str  # of the first expected article
    rank: int | None  # 1-based rank of the first chunk from an expected article
    score: float | None  # that chunk's similarity
    top_titles: tuple[str, ...]


@dataclass
class RetrievalReport:
    k: int
    min_score: float
    results: list[QueryResult]
    off_topic: list[tuple[str, float]] = field(default_factory=list)  # question, best score

    def hit_rate(self, k: int | None = None) -> float:
        k = k or self.k
        return sum(1 for r in self.results if r.rank and r.rank <= k) / len(self.results)

    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)

    def usable_rate(self) -> float:
        """Share of questions whose hit an agent would keep (score at or above min_score)."""
        return sum(1 for r in self.results if r.rank and r.score >= self.min_score) / len(self.results)

    def off_topic_rejection_rate(self) -> float:
        return sum(1 for _, score in self.off_topic if score < self.min_score) / len(self.off_topic)

    def misses(self) -> list[QueryResult]:
        return [r for r in self.results if r.rank is None]

    def by_category(self) -> dict[str, tuple[int, int]]:
        """Category → (hits, questions)."""
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.results:
            counts[r.category][1] += 1
            counts[r.category][0] += bool(r.rank)
        return {c: (h, n) for c, (h, n) in sorted(counts.items())}


def load_golden(path: Path = GOLDEN_SET) -> tuple[list[GoldenQuery], list[str]]:
    data = yaml.safe_load(path.read_text())
    queries = [GoldenQuery(q["question"], tuple(q["expected"])) for q in data["queries"]]
    return queries, list(data.get("off_topic", []))


def article_index(articles_dir: Path = ARTICLES_DIR) -> dict[str, tuple[str, str]]:
    """Article file → (title, category). Retrieved chunks carry the title, not the file name."""
    return {d.metadata["file"]: (d.metadata["title"], d.metadata["category"]) for d in load_articles(articles_dir)}


def first_match(titles: list[str], expected_titles: set[str]) -> int | None:
    return next((i for i, title in enumerate(titles, start=1) if title in expected_titles), None)


def evaluate(retriever: Retriever, queries: list[GoldenQuery], off_topic: list[str],
             articles: dict[str, tuple[str, str]], k: int = 5, min_score: float = 0.0) -> RetrievalReport:
    results = []
    for query in queries:
        docs = retriever.retrieve(query.question, k=k)
        titles = [d.source.title for d in docs]
        rank = first_match(titles, {articles[f][0] for f in query.expected})
        results.append(QueryResult(query, articles[query.expected[0]][1], rank,
                                   docs[rank - 1].score if rank else None, tuple(titles)))
    rejected = []
    for question in off_topic:
        docs = retriever.retrieve(question, k=1)
        rejected.append((question, docs[0].score if docs else 0.0))
    return RetrievalReport(k, min_score, results, rejected)


def format_report(report: RetrievalReport) -> str:
    n = len(report.results)
    lines = [
        f"Retrieval evaluation: {n} questions, top {report.k} chunks, min score {report.min_score:.2f}",
        "",
        *(f"  hit@{k:<2}  {report.hit_rate(k):6.1%}" for k in sorted({1, 3, report.k}) if k <= report.k),
        f"  MRR     {report.mrr():6.3f}",
        f"  usable  {report.usable_rate():6.1%}   (hit@{report.k} with a score an agent keeps)",
        f"  off-topic rejected  {report.off_topic_rejection_rate():6.1%}   ({len(report.off_topic)} questions)",
        "",
        "Score margins around the min score:",
        f"  lowest score of a hit         {min((r.score for r in report.results if r.rank), default=0):.3f}",
        f"  highest off-topic best score  {max((s for _, s in report.off_topic), default=0):.3f}",
        "",
        "By category (hits / questions):",
        *(f"  {category:<20} {hits}/{total}" for category, (hits, total) in report.by_category().items()),
    ]
    if report.misses():
        lines += ["", "Misses:"]
        for r in report.misses():
            lines.append(f"  - {r.query.question}  (expected {', '.join(r.query.expected)})")
            lines.append(f"    got: {' | '.join(r.top_titles[:3])}")
    not_first = [r for r in report.results if r.rank and r.rank > 1]
    if not_first:
        lines += ["", "Found, but not first:"]
        lines += [f"  - {r.query.question}  (rank {r.rank}; first: {r.top_titles[0]})" for r in not_first]
    weak = [r for r in report.results if r.rank and r.score < report.min_score]
    if weak:
        lines += ["", "Found but below the min score (an agent would drop them):"]
        lines += [f"  - {r.query.question}  (score {r.score:.3f})" for r in weak]
    accepted = [(q, s) for q, s in report.off_topic if s >= report.min_score]
    if accepted:
        lines += ["", "Off-topic questions that would get knowledge base context:"]
        lines += [f"  - {q}  (score {s:.3f})" for q, s in accepted]
    return "\n".join(lines)


def main() -> None:
    from src.rag.faiss_retriever import FaissRetriever

    parser = argparse.ArgumentParser(description="Evaluate knowledge base retrieval on the golden set.")
    parser.add_argument("-k", type=int, default=5, help="chunks retrieved per question (default 5, as Finance Q&A)")
    args = parser.parse_args()

    queries, off_topic = load_golden()
    report = evaluate(FaissRetriever.from_settings(), queries, off_topic, article_index(), k=args.k,
                      min_score=min_relevance_score(get_settings()))
    print(format_report(report))


if __name__ == "__main__":
    main()
