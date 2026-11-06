"""Signed-in state for the Streamlit pages: who is logged in, and their profile and portfolio."""

import uuid

import streamlit as st

from src.core.models import Portfolio, UserProfile
from src.data.user_store import Account, UserStore


@st.cache_resource(show_spinner=False)
def get_user_store() -> UserStore:
    """One MongoDB connection per process. Raises UserStoreUnavailable (not cached) if MongoDB is down."""
    return UserStore.from_settings()


def start_new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("goal_question", None)


def init_session() -> None:
    if "session_id" not in st.session_state:
        start_new_conversation()


def current_account() -> Account | None:
    return st.session_state.get("account")


def log_in(store: UserStore, account: Account) -> None:
    """Remember the account and load its profile and portfolio; starts a fresh conversation."""
    start_new_conversation()
    st.session_state.account = account
    st.session_state.profile = store.load_profile(account.id, st.session_state.session_id)
    st.session_state.portfolio = store.load_portfolio(account.id)


def log_out() -> None:
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
