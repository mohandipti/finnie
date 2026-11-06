"""Browser sessions and chat transcripts stored in MongoDB, so a page refresh keeps the user's place.

Streamlit forgets everything when the page is refreshed, because each refresh is a new connection.
The server gives each browser a random session cookie (see web_app/server.py); these collections
map that cookie back to the signed-in account and the conversation on screen:

    browser_sessions  _id = SHA-256 of the cookie, user_id, conversation_id, expires_at
    transcripts       _id = conversation_id, user_id, messages[], expires_at

Only a hash of the cookie is stored, so the database alone can't be used to take over a session.
Both collections expire documents through a TTL index on expires_at.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from src.core.config import Settings, get_settings
from src.core.models import Source
from src.data.user_store import UserStoreUnavailable

logger = logging.getLogger(__name__)

SESSION_COOKIE = "finnie_session"


@dataclass(frozen=True)
class BrowserSession:
    user_id: str
    conversation_id: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


class SessionStore:
    def __init__(self, db: Database, session_ttl: timedelta, transcript_ttl: timedelta) -> None:
        self.sessions = db["browser_sessions"]
        self.transcripts = db["transcripts"]
        self.session_ttl = session_ttl
        self.transcript_ttl = transcript_ttl

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "SessionStore":
        """Connect to MONGODB_URI and create the indexes; raises UserStoreUnavailable if MongoDB is down."""
        settings = settings or get_settings()
        try:
            client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=3000, tz_aware=True)
            client.admin.command("ping")
            store = cls(client[settings.mongodb_db], timedelta(days=settings.session_ttl_days),
                        timedelta(days=settings.conversation_ttl_days))
            store.ensure_indexes()
        except PyMongoError as exc:
            # Log the error type only: the message can include the connection string.
            logger.error("MongoDB is unavailable: %s", type(exc).__name__)
            raise UserStoreUnavailable("Sessions are unavailable right now: Finnie can't reach its database.") from exc
        return store

    def ensure_indexes(self) -> None:
        self.sessions.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
        self.transcripts.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

    def load(self, token: str) -> BrowserSession | None:
        """The session for this cookie, or None if it's unknown or expired. Each use extends it."""
        now = _now()
        doc = self.sessions.find_one_and_update(
            {"_id": _hash(token), "expires_at": {"$gt": now}},
            {"$set": {"expires_at": now + self.session_ttl}},
        )
        return BrowserSession(str(doc["user_id"]), doc["conversation_id"]) if doc else None

    def save(self, token: str, user_id: str, conversation_id: str) -> None:
        self.sessions.update_one(
            {"_id": _hash(token)},
            {"$set": {"user_id": ObjectId(user_id), "conversation_id": conversation_id,
                      "expires_at": _now() + self.session_ttl}},
            upsert=True,
        )

    def delete(self, token: str) -> None:
        self.sessions.delete_one({"_id": _hash(token)})

    def load_transcript(self, conversation_id: str, user_id: str) -> list[dict]:
        doc = self.transcripts.find_one({"_id": conversation_id, "user_id": ObjectId(user_id)})
        if doc is None:
            return []
        return [{**m, "sources": [Source.model_validate(s) for s in m.get("sources", [])]} for m in doc["messages"]]

    def save_transcript(self, conversation_id: str, user_id: str, messages: list[dict]) -> None:
        stored = [{**m, "sources": [s.model_dump() for s in m.get("sources", [])]} for m in messages]
        self.transcripts.update_one(
            {"_id": conversation_id},
            {"$set": {"user_id": ObjectId(user_id), "messages": stored,
                      "expires_at": _now() + self.transcript_ttl}},
            upsert=True,
        )
