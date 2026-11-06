import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from src.core.config import Settings
from src.workflow import graph as graph_module
from src.workflow.graph import default_checkpointer
from src.workflow.state import FinnieState


class DownClient:
    def __init__(self, *args, **kwargs):
        self.admin = self

    def command(self, name):
        raise ServerSelectionTimeoutError("no servers")


def test_memory_store_can_be_chosen():
    assert isinstance(default_checkpointer(Settings(conversation_store="memory")), InMemorySaver)


def test_falls_back_to_memory_when_mongodb_is_down(monkeypatch):
    monkeypatch.setattr(graph_module, "MongoClient", DownClient)
    assert isinstance(default_checkpointer(Settings()), InMemorySaver)


def test_mongodb_saver_uses_the_finnie_database_and_expiry(monkeypatch):
    created = {}
    monkeypatch.setattr(graph_module, "MongoClient", lambda *a, **k: type("Up", (), {"admin": DownClient()})())
    monkeypatch.setattr(DownClient, "command", lambda self, name: {"ok": 1})
    monkeypatch.setattr(graph_module, "MongoDBSaver", lambda client, **kwargs: created.update(kwargs) or "saver")

    assert default_checkpointer(Settings(mongodb_db="finnie_x", conversation_ttl_days=7)) == "saver"
    assert created["db_name"] == "finnie_x"
    assert created["checkpoint_collection_name"] == "conversation_checkpoints"
    assert created["ttl"] == 7 * 24 * 3600


@pytest.fixture
def throwaway_settings():
    """Settings pointing at a temporary database on the local MongoDB, dropped afterwards."""
    settings = Settings(mongodb_db=f"finnie_test_{uuid.uuid4().hex[:8]}")
    client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=500)
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip("MongoDB is not running")
    yield settings
    client.drop_database(settings.mongodb_db)


def test_paused_turn_resumes_from_another_instance(throwaway_settings):
    def build():
        g = StateGraph(FinnieState)
        g.add_node("ask", lambda s: {"final_response": f"counted {interrupt({'q': 'how much?'})['amount']}"})
        g.add_edge(START, "ask")
        g.add_edge("ask", END)
        return g.compile(checkpointer=default_checkpointer(throwaway_settings))

    config = {"configurable": {"thread_id": "s1"}}
    first = build()
    first.invoke({"user_query": "hi"}, config)
    assert first.get_state(config).interrupts

    second = build()  # e.g. after a restart or on another ECS task
    assert second.get_state(config).interrupts[0].value == {"q": "how much?"}
    assert second.invoke(Command(resume={"amount": 5}), config)["final_response"] == "counted 5"
