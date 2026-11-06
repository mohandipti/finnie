"""Tax Education Agent: US tax concepts and tax-advantaged accounts, grounded in the knowledge base.

Works like the Finance Q&A Agent with a tax-specific prompt. Retrieval searches the
tax and retirement articles first, then the whole knowledge base if none match,
since some tax topics (municipal bonds, I bonds) live in other categories.
"""

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.prompts import TAX_SYSTEM_PROMPT
from src.core.models import RetrievedDocument
from src.workflow.state import FinnieState


class TaxAgent(FinanceQAAgent):
    name = "tax"
    description = "US tax concepts and account types such as IRAs, 401(k)s, HSAs and capital gains"
    system_prompt = TAX_SYSTEM_PROMPT
    retrieval_categories = ["taxes", "retirement"]  # noqa: RUF012 - read-only

    def retrieve_context(self, state: FinnieState) -> list[RetrievedDocument]:
        query = self.retrieval_query(state)
        return self.retrieve(query, categories=self.retrieval_categories) or self.retrieve(query)
