import uuid
from datetime import UTC, datetime, timedelta

import pytest
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.core.config import Settings
from src.core.models import Source
from src.data.session_store import SESSION_COOKIE, SessionStore
from src.web_app.server import SessionCookieMiddleware

USER_ID = str(ObjectId())


@pytest.fixture
def store():
    """A SessionStore on a temporary database on the local MongoDB, dropped afterwards."""
    settings = Settings(mongodb_db=f"finnie_test_{uuid.uuid4().hex[:8]}")
    client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=500)
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip("MongoDB is not running")
    yield SessionStore.from_settings(settings)
    client.drop_database(settings.mongodb_db)


def test_saved_session_is_found_by_its_cookie_only(store):
    store.save("token-a", USER_ID, "conv-1")

    assert store.load("token-a").conversation_id == "conv-1"
    assert store.load("token-b") is None
    assert store.sessions.find_one({"_id": "token-a"}) is None  # only the hash is stored


def test_logging_out_deletes_the_session(store):
    store.save("token-a", USER_ID, "conv-1")
    store.delete("token-a")
    assert store.load("token-a") is None


def test_expired_session_is_not_restored(store):
    store.save("token-a", USER_ID, "conv-1")
    store.sessions.update_many({}, {"$set": {"expires_at": datetime.now(UTC) - timedelta(minutes=1)}})
    assert store.load("token-a") is None


def test_transcript_round_trip_keeps_sources_and_belongs_to_its_user(store):
    messages = [
        {"role": "user", "content": "What is an ETF?"},
        {"role": "assistant", "content": "A basket of investments.", "agents": ["finance_qa"],
         "sources": [Source(title="ETFs", url="https://investor.gov", kind="kb")]},
    ]
    store.save_transcript("conv-1", USER_ID, messages)

    restored = store.load_transcript("conv-1", USER_ID)
    assert restored[1]["sources"][0] == Source(title="ETFs", url="https://investor.gov", kind="kb")
    assert restored[0]["content"] == "What is an ETF?"
    assert store.load_transcript("conv-1", str(ObjectId())) == []


def cookie_client() -> TestClient:
    app = Starlette(routes=[Route("/{path:path}", lambda request: PlainTextResponse("ok"))])
    app.add_middleware(SessionCookieMiddleware)
    return TestClient(app)


def test_page_load_gets_an_httponly_session_cookie():
    response = cookie_client().get("/assistant")
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie


def test_cookie_is_not_replaced_or_set_on_internal_routes():
    client = cookie_client()
    assert "set-cookie" not in client.get("/", cookies={SESSION_COOKIE: "existing"}).headers
    assert "set-cookie" not in cookie_client().get("/_stcore/health").headers
