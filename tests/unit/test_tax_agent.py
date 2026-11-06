from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents.tax_agent import TaxAgent
from src.core.models import RetrievedDocument, Source
from src.workflow.router import keyword_route

HSA_DOC = RetrievedDocument(content="HSA contributions are deductible.", score=0.8,
                            source=Source(title="Health Savings Accounts (HSAs)", category="taxes", kind="kb"))
MUNI_DOC = RetrievedDocument(content="Muni bond interest is often exempt from federal tax.", score=0.75,
                             source=Source(title="Municipal Bonds", category="bonds", kind="kb"))


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def retrieve(self, query, k=5, categories=None):
        self.calls.append(categories)
        return [d for d in self.docs if categories is None or d.source.category in categories]


def agent(docs, *replies):
    retriever = FakeRetriever(docs)
    llm = GenericFakeChatModel(messages=iter(replies or [AIMessage(content="answer")]))
    return TaxAgent(llm=llm, retriever=retriever), retriever


def state(query):
    return {"session_id": "s1", "user_query": query, "messages": []}


def test_searches_tax_and_retirement_articles_first():
    tax, retriever = agent([HSA_DOC, MUNI_DOC])
    docs = tax.retrieve_context(state("Are HSA contributions deductible?"))

    assert docs == [HSA_DOC]
    assert retriever.calls == [["taxes", "retirement"]]


def test_falls_back_to_the_whole_knowledge_base():
    tax, retriever = agent([MUNI_DOC])
    docs = tax.retrieve_context(state("Is municipal bond interest taxed?"))

    assert docs == [MUNI_DOC]
    assert retriever.calls == [["taxes", "retirement"], None]


def test_answer_cites_retrieved_articles():
    tax, _ = agent([HSA_DOC], AIMessage(content="Yes, HSA contributions are deductible [1]."))
    response = tax.run(state("Are HSA contributions deductible?"))

    assert response.error is None and response.agent == "tax"
    assert [s.title for s in response.sources] == ["Health Savings Accounts (HSAs)"]


def test_keyword_fallback_routes_tax_questions_to_tax():
    decision = keyword_route("How are capital gains taxed?", ["finance_qa", "tax"])
    assert decision.agents == ["tax"]
