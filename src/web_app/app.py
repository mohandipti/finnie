"""Finnie Streamlit app: page setup and navigation. Started through server.py (`finnie`), which adds the session cookie."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allows `streamlit run` without installing the package
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.web_app.auth import init_session

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(page_title="Finnie", page_icon="💬", layout="wide")
init_session()

home = st.Page(PAGES_DIR / "home.py", title="Home", icon=":material/home:", default=True)
# Reached from the links on the home page, not the menu.
assistant = st.Page(PAGES_DIR / "assistant.py", title="Assistant", icon=":material/forum:", url_path="assistant",
                    visibility="hidden")
signup = st.Page(PAGES_DIR / "signup.py", title="Sign up", icon=":material/person_add:", url_path="signup",
                 visibility="hidden")
account = st.Page(PAGES_DIR / "account.py", title="My account", icon=":material/account_circle:",
                  url_path="account", visibility="hidden")
login = st.Page(PAGES_DIR / "login.py", title="Log in", icon=":material/login:", url_path="login",
                visibility="hidden")

st.navigation([home, assistant, signup, login, account], position="top").run()
