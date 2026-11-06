"""Sign-up page: account, profile, goals and portfolio, saved to MongoDB."""

import streamlit as st
from pydantic import ValidationError

from src.core.models import Portfolio, UserProfile
from src.data.user_store import AccountError, NewAccount, UserStoreUnavailable
from src.web_app.auth import current_account, get_user_store, log_in
from src.web_app.components.profile_inputs import about_you_inputs, cash_input, goals_editor, holdings_editor
from src.web_app.components.signup_forms import (
    cash_holding,
    empty_goals,
    empty_holdings,
    parse_goals,
    parse_holdings,
)

st.title("Create your Finnie account", anchor=False)

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

st.caption("Finnie uses these details to tailor its explanations. Only the account section is required.")

with st.container(border=True):
    st.subheader(":material/badge: Account", anchor=False)
    name = st.text_input("Name", max_chars=100)
    email = st.text_input("Email")
    left, right = st.columns(2)
    password = left.text_input("Password", type="password", help="At least 8 characters.")
    confirm = right.text_input("Confirm password", type="password")

with st.container(border=True):
    st.subheader(":material/person: About you", anchor=False)
    level, risk, interests = about_you_inputs()

with st.container(border=True):
    st.subheader(":material/flag: Goals", anchor=False)
    st.caption("Optional. Add a row per goal, such as a house deposit or retirement. "
               "Leave Risk blank to use your answer above.")
    goals_df = goals_editor(empty_goals(), key="signup_goals")

with st.container(border=True):
    st.subheader(":material/pie_chart: Portfolio", anchor=False)
    st.caption("Optional. Finnie supports US-listed stocks, ETFs and funds only.")
    holdings_df = holdings_editor(empty_holdings(), key="signup_holdings")
    cash = cash_input()

if not st.button("Create account", type="primary", icon=":material/person_add:"):
    st.page_link("app_pages/login.py", label="Already have an account? Log in", icon=":material/login:")
    st.stop()

# Validate everything first, so nothing is saved unless the whole form is valid.
errors: list[str] = []
new_account = None
try:
    new_account = NewAccount(name=name, email=email, password=password)
except ValidationError as exc:
    for error in exc.errors():
        errors.append(f"{str(error['loc'][0]).title()}: {error['msg'].removeprefix('Value error, ')}")
if password != confirm:
    errors.append("Password: the two passwords don't match")

goals, goal_errors = parse_goals(goals_df, risk)
errors += goal_errors

holdings, holding_errors = parse_holdings(holdings_df)
errors += holding_errors
if cash_position := cash_holding(cash):
    holdings.append(cash_position)

if errors:
    st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in errors), icon=":material/error:")
    st.stop()

profile = UserProfile(session_id=st.session_state.session_id, knowledge_level=level, risk_tolerance=risk,
                      goals=goals, interests=interests)
portfolio = Portfolio(holdings=holdings) if holdings else None
try:
    created = store.create_account(new_account, profile, portfolio)
except AccountError as exc:
    st.error(str(exc), icon=":material/error:")
    st.page_link("app_pages/login.py", label="Log in", icon=":material/login:")
    st.stop()

log_in(store, created)
st.switch_page("app_pages/assistant.py")
