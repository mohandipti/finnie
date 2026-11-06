"""Assistant page: the chat, portfolio and market tabs, with the profile settings in the sidebar.

Signed-in users only; anyone else is sent to the log-in page.
"""

import streamlit as st

from src.core.disclaimers import DISCLAIMER
from src.core.models import UserProfile
from src.data.user_store import UserStoreUnavailable
from src.utils.logger import setup_logger
from src.web_app.auth import (
    current_account,
    current_portfolio,
    current_profile,
    get_user_store,
    log_out,
    start_new_conversation,
)
from src.web_app.tabs import chat, market, portfolio

if current_account() is None:
    st.switch_page("app_pages/login.py")

KNOWLEDGE_LEVELS = ["beginner", "intermediate", "advanced"]
RISK_LEVELS = ["conservative", "moderate", "aggressive"]


@st.cache_resource(show_spinner="Loading Finnie...")
def get_graph():
    """Build the workflow once per process: it holds the FAISS index, LLM clients and conversation memory."""
    from src.workflow.graph import FinnieGraph

    setup_logger()
    return FinnieGraph()


def save_profile_changes(profile: UserProfile) -> None:
    """Keep the signed-in user's sidebar choices in their stored profile."""
    stored = st.session_state.get("profile")
    if stored is None:
        return
    if (profile.knowledge_level, profile.risk_tolerance) == (stored.knowledge_level, stored.risk_tolerance):
        return
    try:
        get_user_store().save_profile(current_account().id, profile)
    except UserStoreUnavailable as exc:
        st.toast(str(exc), icon=":material/error:")
        return
    st.session_state.profile = profile


def sidebar() -> UserProfile:
    base = current_profile()
    account = current_account()
    with st.sidebar:
        st.header("Your profile")
        level = st.selectbox("Knowledge level", KNOWLEDGE_LEVELS, index=KNOWLEDGE_LEVELS.index(base.knowledge_level),
                             format_func=str.title)
        risk = st.selectbox("Risk tolerance", RISK_LEVELS, index=RISK_LEVELS.index(base.risk_tolerance or "moderate"),
                            format_func=str.title)
        st.caption(f"Signed in as {account.name}. Changes are saved to your profile.")
        # All actions are full-width buttons with icons so they read as one group. Page changes
        # happen inline because switch_page can't be called from an on_click callback.
        st.button("New conversation", on_click=start_new_conversation, icon=":material/add_comment:",
                  width="stretch")
        if st.button("My account", icon=":material/account_circle:", width="stretch"):
            st.switch_page("app_pages/account.py")
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            log_out()
            st.switch_page("app_pages/home.py")
        st.divider()
        st.caption(DISCLAIMER)
    return base.model_copy(update={"knowledge_level": level, "risk_tolerance": risk})


st.title("Finnie")
st.caption("Your AI guide to personal finance and investing basics")
profile = sidebar()
save_profile_changes(profile)

graph, error = None, None
try:
    graph = get_graph()
except Exception as exc:  # noqa: BLE001 - show setup problems (missing key or index) in the UI
    error = f"Finnie couldn't start: {exc}"

# on_change="rerun" tracks the selected tab, so the Portfolio and Market tabs only fetch prices while open.
chat_tab, portfolio_tab, market_tab = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Market"], key="assistant_tab",
                                              on_change="rerun")
with chat_tab:
    chat.render(graph, profile, current_portfolio(), error)
if portfolio_tab.open:
    with portfolio_tab:
        portfolio.render(current_portfolio())
if market_tab.open:
    with market_tab:
        market.render(current_portfolio())
