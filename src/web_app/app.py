"""Finnie Streamlit app. Run with `finnie` or `streamlit run src/web_app/app.py`."""

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allows `streamlit run` without installing the package
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.disclaimers import DISCLAIMER
from src.core.models import UserProfile
from src.data.sample_data import load_sample_profile
from src.utils.logger import setup_logger
from src.web_app.tabs import chat, market, portfolio

KNOWLEDGE_LEVELS = ["beginner", "intermediate", "advanced"]
RISK_LEVELS = ["conservative", "moderate", "aggressive"]


@st.cache_resource(show_spinner="Loading Finnie...")
def get_graph():
    """Build the workflow once per process: it holds the FAISS index, LLM clients and conversation memory."""
    from src.workflow.graph import FinnieGraph

    setup_logger()
    return FinnieGraph()


def start_new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def sidebar() -> UserProfile:
    base = load_sample_profile(st.session_state.session_id)
    with st.sidebar:
        st.header("Your profile")
        level = st.selectbox("Knowledge level", KNOWLEDGE_LEVELS, index=KNOWLEDGE_LEVELS.index(base.knowledge_level),
                             format_func=str.title)
        risk = st.selectbox("Risk tolerance", RISK_LEVELS, index=RISK_LEVELS.index(base.risk_tolerance or "moderate"),
                            format_func=str.title)
        st.caption("Sample profile for now; sign-up is coming.")
        st.button("New conversation", on_click=start_new_conversation, use_container_width=True)
        st.divider()
        st.caption(DISCLAIMER)
    return base.model_copy(update={"knowledge_level": level, "risk_tolerance": risk})


def main() -> None:
    st.set_page_config(page_title="Finnie", page_icon="💬", layout="wide", initial_sidebar_state="expanded")
    if "session_id" not in st.session_state:
        start_new_conversation()

    st.title("Finnie")
    st.caption("Your AI guide to personal finance and investing basics")
    profile = sidebar()

    graph, error = None, None
    try:
        graph = get_graph()
    except Exception as exc:  # noqa: BLE001 - show setup problems (missing key or index) in the UI
        error = f"Finnie couldn't start: {exc}"

    chat_tab, portfolio_tab, market_tab = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Market"])
    with chat_tab:
        chat.render(graph, profile, error)
    with portfolio_tab:
        portfolio.render()
    with market_tab:
        market.render()


main()
