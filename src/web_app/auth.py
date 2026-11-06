"""Signed-in state for the Streamlit pages: who is logged in, and their profile and portfolio.

Streamlit's session state is lost when the page is refreshed, so the signed-in account and the
conversation on screen are also saved against the browser's session cookie (see server.py) and
restored from MongoDB when a refreshed page starts a new session.
"""

import logging
import uuid

import streamlit as st
from pymongo.errors import PyMongoError

from src.core.models import Portfolio, UserProfile
from src.data.session_store import SESSION_COOKIE, SessionStore
from src.data.user_store import Account, UserStore, UserStoreUnavailable

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_user_store() -> UserStore:
    """One MongoDB connection per process. Raises UserStoreUnavailable (not cached) if MongoDB is down."""
    return UserStore.from_settings()


@st.cache_resource(show_spinner=False)
def get_session_store() -> SessionStore:
    """Raises UserStoreUnavailable (not cached) if MongoDB is down."""
    return SessionStore.from_settings()


def _session_token() -> str | None:
    return st.context.cookies.get(SESSION_COOKIE)


def _saved_session() -> tuple[SessionStore, str] | None:
    """The session store and this browser's cookie, if both are available."""
    token = _session_token()
    if not token:
        return None
    try:
        return get_session_store(), token
    except UserStoreUnavailable:
        return None


def _remember() -> None:
    """Point this browser's saved session at the signed-in account and current conversation."""
    account, saved = current_account(), _saved_session()
    if account is None or saved is None:
        return
    store, token = saved
    try:
        store.save(token, account.id, st.session_state.session_id)
    except PyMongoError as exc:
        logger.warning("Couldn't save the browser session: %s", type(exc).__name__)


def save_transcript() -> None:
    """Save the conversation on screen, so a page refresh shows it again."""
    account, saved = current_account(), _saved_session()
    if account is None or saved is None:
        return
    try:
        saved[0].save_transcript(st.session_state.session_id, account.id, st.session_state.messages)
    except PyMongoError as exc:
        logger.warning("Couldn't save the transcript: %s", type(exc).__name__)


def _restore_session() -> bool:
    """Sign the browser back in after a page refresh, with the conversation it had open."""
    saved = _saved_session()
    if saved is None:
        return False
    store, token = saved
    try:
        session = store.load(token)
        if session is None:
            return False
        user_store = get_user_store()
        account = user_store.get_account(session.user_id)
        if account is None:
            return False
        st.session_state.session_id = session.conversation_id
        st.session_state.messages = store.load_transcript(session.conversation_id, account.id)
        _load_account(user_store, account)
    except (PyMongoError, UserStoreUnavailable) as exc:
        logger.warning("Couldn't restore the browser session: %s", type(exc).__name__)
        return False
    st.session_state.restored = True  # the chat checks whether the turn was paused on a question
    return True


def start_new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("goal_question", None)
    _remember()


def init_session() -> None:
    if "session_id" not in st.session_state and not _restore_session():
        start_new_conversation()


def current_account() -> Account | None:
    return st.session_state.get("account")


def _load_account(store: UserStore, account: Account) -> None:
    st.session_state.account = account
    st.session_state.profile = store.load_profile(account.id, st.session_state.session_id)
    st.session_state.portfolio = store.load_portfolio(account.id)


def log_in(store: UserStore, account: Account) -> None:
    """Remember the account and load its profile and portfolio; starts a fresh conversation."""
    start_new_conversation()
    _load_account(store, account)
    _remember()


def log_out() -> None:
    saved = _saved_session()
    if saved is not None:
        try:
            saved[0].delete(saved[1])
        except PyMongoError as exc:
            logger.warning("Couldn't delete the browser session: %s", type(exc).__name__)
    for key in ("account", "profile", "portfolio"):
        st.session_state.pop(key, None)
    start_new_conversation()


def current_profile() -> UserProfile:
    """The signed-in user's profile, or a blank one if theirs is missing."""
    profile = st.session_state.get("profile") if current_account() else None
    if profile is None:
        return UserProfile(session_id=st.session_state.session_id)
    return profile.model_copy(update={"session_id": st.session_state.session_id})


def current_portfolio() -> Portfolio | None:
    return st.session_state.get("portfolio") if current_account() else None
