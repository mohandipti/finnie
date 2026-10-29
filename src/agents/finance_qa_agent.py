"""Finance Q&A Agent: general financial education grounded in the knowledge base."""

import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.prompts import FINANCE_QA_SYSTEM_PROMPT
from src.core.models import RetrievedDocument
from src.workflow.state import FinnieState

NO_CONTEXT_NOTE = (
    "No knowledge base articles matched this question. Tell the user that Finnie's knowledge base "
    "doesn't cover it directly, then give a brief, hedged general explanation based on widely "
    "accepted principles, or say you can't help if it isn't a personal finance topic."
)

FOLLOW_UP_MAX_WORDS = 8
# A short query is treated as a follow-up only if it refers back to something ("what about its fees?"),
# so an unrelated short question ("best pizza recipe") isn't searched with the previous topic.
FOLLOW_UP_MARKERS = re.compile(
    r"^(and|also|what about|how about|why|so)\b|\b(it|its|it's|they|them|their|this|that|these|those|one)\b",
    re.IGNORECASE,
)


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    description = "General financial concepts, terminology and how investing works"
    system_prompt = FINANCE_QA_SYSTEM_PROMPT

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        return self.retrieve(self.retrieval_query(state), categories=self.retrieval_categories)

    def retrieval_query(self, state: FinnieState) -> str:
        """The search query; short follow-ups are prefixed with the previous question so retrieval has context."""
        query = state.get("user_query", "").strip()
        if len(query.split()) > FOLLOW_UP_MAX_WORDS or not FOLLOW_UP_MARKERS.search(query):
            return query
        previous = [m for m in self.recent_history(state) if m.type == "human"]
        return f"{previous[-1].text} {query}" if previous else query

    def build_user_message(
        self, state: FinnieState, docs: list[RetrievedDocument], data: dict[str, Any] | None
    ) -> str:
        query = state.get("user_query", "")
        if not docs:
            return f"{NO_CONTEXT_NOTE}\n\nQuestion: {query}"
        return (
            "Knowledge base excerpts (cite them by number, e.g. [1]):\n\n"
            f"{self.format_context(docs)}\n\n"
            f"Question: {query}"
        )
