"""Log-in page: email and password, checked against MongoDB."""

import streamlit as st

from src.data.user_store import UserStoreUnavailable
from src.web_app.auth import current_account, get_user_store, log_in

st.title("Log in", anchor=False)

account = current_account()
if account:
    st.success(f"You're signed in as {account.name} ({account.email}).", icon=":material/check_circle:")
    st.page_link("app_pages/assistant.py", label="Open Finnie", icon=":material/forum:")
    st.stop()

try:
    store = get_user_store()
except UserStoreUnavailable as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

with st.form("login", width=480):
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Log in", type="primary", icon=":material/login:")

if submitted:
    found = store.authenticate(email, password)
    if found is None:
        st.error("That email and password don't match an account.", icon=":material/error:")
    else:
        log_in(store, found)
        st.switch_page("app_pages/assistant.py")

st.page_link("app_pages/signup.py", label="New to Finnie? Create an account", icon=":material/person_add:")
