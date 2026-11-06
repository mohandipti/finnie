"""User accounts, profiles and portfolios stored in MongoDB.

Three collections, one document each per user:

    users       email (unique), name, password_hash, created_at, last_login_at
    profiles    user_id (unique), knowledge_level, risk_tolerance, interests[], goals[], updated_at
    portfolios  user_id (unique), market, holdings[], updated_at

Goals and holdings are embedded in their parent document: they are small, always read
together and saved in one write. Passwords are stored only as bcrypt hashes.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import bcrypt
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from src.core.config import Settings, get_settings
from src.core.models import Portfolio, UserProfile

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt ignores (newer versions reject) anything longer
# Compared against when the email is unknown, so a failed login takes as long either way.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt())


class AccountError(ValueError):
    """A problem the user can fix, such as an email that is already registered."""


class UserStoreUnavailable(RuntimeError):
    """MongoDB can't be reached or isn't configured."""


@dataclass(frozen=True)
class Account:
    """A signed-in user, as kept in the session."""

    id: str
    email: str
    name: str


class NewAccount(BaseModel):
    """Sign-up details, validated before anything is written."""

    name: str = Field(min_length=1, max_length=100)
    email: str
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("enter your name")
        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL_PATTERN.match(value):
            raise ValueError("enter a valid email address")
        return value

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        if len(value.encode()) > MAX_PASSWORD_BYTES:
            raise ValueError(f"use at most {MAX_PASSWORD_BYTES} characters")
        return value


def _now() -> datetime:
    return datetime.now(UTC)


class UserStore:
    def __init__(self, db: Database) -> None:
        self.users = db["users"]
        self.profiles = db["profiles"]
        self.portfolios = db["portfolios"]

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "UserStore":
        """Connect to MONGODB_URI and create the indexes; raises UserStoreUnavailable if MongoDB is down."""
        settings = settings or get_settings()
        try:
            client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=3000, tz_aware=True)
            client.admin.command("ping")
            store = cls(client[settings.mongodb_db])
            store.ensure_indexes()
        except PyMongoError as exc:
            # Log the error type only: the message can include the connection string.
            logger.error("MongoDB is unavailable: %s", type(exc).__name__)
            raise UserStoreUnavailable("Accounts are unavailable right now: Finnie can't reach its database.") from exc
        return store

    def ensure_indexes(self) -> None:
        self.users.create_index([("email", ASCENDING)], unique=True)
        self.profiles.create_index([("user_id", ASCENDING)], unique=True)
        self.portfolios.create_index([("user_id", ASCENDING)], unique=True)

    def create_account(self, new: NewAccount, profile: UserProfile, portfolio: Portfolio | None = None) -> Account:
        """Create the user with their profile and, optionally, portfolio. Raises AccountError if the email is taken."""
        now = _now()
        password_hash = bcrypt.hashpw(new.password.encode(), bcrypt.gensalt()).decode()
        try:
            user_id = self.users.insert_one({
                "email": new.email, "name": new.name, "password_hash": password_hash,
                "created_at": now, "last_login_at": now,
            }).inserted_id
        except DuplicateKeyError as exc:
            raise AccountError("An account with this email already exists. Log in instead.") from exc

        # A standalone MongoDB has no multi-document transactions, so undo the user if the rest fails.
        try:
            self.save_profile(user_id, profile)
            if portfolio is not None:
                self.save_portfolio(user_id, portfolio)
        except Exception:
            self.profiles.delete_one({"user_id": user_id})
            self.portfolios.delete_one({"user_id": user_id})
            self.users.delete_one({"_id": user_id})
            raise
        logger.info("created account %s", user_id)
        return Account(id=str(user_id), email=new.email, name=new.name)

    def authenticate(self, email: str, password: str) -> Account | None:
        """The account for these credentials, or None if the email or password is wrong."""
        user = self.users.find_one({"email": email.strip().lower()})
        stored = user["password_hash"].encode() if user else _DUMMY_HASH
        try:
            matches = bcrypt.checkpw(password.encode(), stored)
        except ValueError:  # e.g. a password longer than bcrypt accepts
            matches = False
        if not (user and matches):
            return None
        self.users.update_one({"_id": user["_id"]}, {"$set": {"last_login_at": _now()}})
        return Account(id=str(user["_id"]), email=user["email"], name=user["name"])

    def get_account(self, user_id: str | ObjectId) -> Account | None:
        user = self.users.find_one({"_id": ObjectId(user_id)})
        return Account(id=str(user["_id"]), email=user["email"], name=user["name"]) if user else None

    def update_name(self, user_id: str | ObjectId, name: str) -> None:
        self.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"name": name}})

    def load_profile(self, user_id: str | ObjectId, session_id: str) -> UserProfile | None:
        doc = self.profiles.find_one({"user_id": ObjectId(user_id)})
        if doc is None:
            return None
        return UserProfile.model_validate({**doc, "session_id": session_id})

    def save_profile(self, user_id: str | ObjectId, profile: UserProfile) -> None:
        fields = profile.model_dump(exclude={"session_id"})
        self.profiles.update_one(
            {"user_id": ObjectId(user_id)}, {"$set": {**fields, "updated_at": _now()}}, upsert=True
        )

    def load_portfolio(self, user_id: str | ObjectId) -> Portfolio | None:
        doc = self.portfolios.find_one({"user_id": ObjectId(user_id)})
        return Portfolio.model_validate(doc) if doc else None

    def save_portfolio(self, user_id: str | ObjectId, portfolio: Portfolio) -> None:
        self.portfolios.update_one(
            {"user_id": ObjectId(user_id)}, {"$set": {**portfolio.model_dump(), "updated_at": _now()}}, upsert=True
        )

    def delete_portfolio(self, user_id: str | ObjectId) -> None:
        self.portfolios.delete_one({"user_id": ObjectId(user_id)})
