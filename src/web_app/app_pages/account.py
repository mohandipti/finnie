"""My account page: view and update the signed-in user's details, goals and portfolio."""

import streamlit as st

from src.core.models import Portfolio
from src.data.user_store import Account, UserStoreUnavailable
from src.web_app.auth import current_account, current_portfolio, current_profile, get_user_store
from src.web_app.components.profile_inputs import about_you_inputs, cash_input, goals_editor, holdings_editor
from src.web_app.components.signup_forms import (
    cash_holding,
    goals_frame,
    holdings_frame,
    parse_goals,
    parse_holdings,
)


def show_errors(errors: list[str]) -> None:
    st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in errors), icon=":material/error:")


def saved(message: str) -> None:
    # A new version gives the table editors fresh keys, so they show the saved rows rather than
    # replaying the previous edits on top of them.
    st.session_state.account_version = st.session_state.get("account_version", 0) + 1
    st.toast(message, icon=":material/check_circle:")
    st.rerun()


st.title("My account", anchor=False)

account = current_account()
if account is None:
    st.info("Log in to see your account.", icon=":material/lock:")
    st.page_link("app_pages/login.py", label="Log in", icon=":material/login:")
    st.stop()

try:
    store = get_user_store()
except UserStoreUnavailable as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

profile = current_profile()
portfolio = current_portfolio()
version = st.session_state.get("account_version", 0)
st.caption(f"Signed in as {account.email}. Each section is saved separately.")

with st.form("account_details", border=True):
    st.subheader(":material/tune: Details and settings", anchor=False)
    name = st.text_input("Name", value=account.name, max_chars=100)
    level, risk, interests = about_you_inputs(profile.knowledge_level, profile.risk_tolerance or "moderate",
                                              profile.interests)
    if st.form_submit_button("Save details", type="primary", icon=":material/save:"):
        if not name.strip():
            show_errors(["Name: enter your name"])
        else:
            updated = profile.model_copy(update={"knowledge_level": level, "risk_tolerance": risk,
                                                 "interests": interests})
            store.save_profile(account.id, updated)
            if name.strip() != account.name:
                store.update_name(account.id, name.strip())
                st.session_state.account = Account(id=account.id, email=account.email, name=name.strip())
            st.session_state.profile = updated
            saved("Details saved")

with st.form("account_goals", border=True):
    st.subheader(":material/flag: Goals", anchor=False)
    st.caption("Add, change or delete rows. Leave Risk blank to use your risk tolerance.")
    goals_df = goals_editor(goals_frame(profile.goals), key=f"account_goals_{version}")
    if st.form_submit_button("Save goals", type="primary", icon=":material/save:"):
        goals, errors = parse_goals(goals_df, profile.risk_tolerance or "moderate")
        if errors:
            show_errors(errors)
        else:
            updated = profile.model_copy(update={"goals": goals})
            store.save_profile(account.id, updated)
            st.session_state.profile = updated
            saved("Goals saved")

with st.form("account_portfolio", border=True):
    st.subheader(":material/pie_chart: Portfolio", anchor=False)
    st.caption("US-listed stocks, ETFs and funds only. Clear every row and set cash to 0 to remove the portfolio.")
    holdings_data, cash_amount = holdings_frame(portfolio)
    holdings_df = holdings_editor(holdings_data, key=f"account_holdings_{version}")
    cash = cash_input(cash_amount, key=f"account_cash_{version}")
    if st.form_submit_button("Save portfolio", type="primary", icon=":material/save:"):
        holdings, errors = parse_holdings(holdings_df)
        if errors:
            show_errors(errors)
        else:
            if cash_position := cash_holding(cash):
                holdings.append(cash_position)
            if holdings:
                updated_portfolio = Portfolio(holdings=holdings)
                store.save_portfolio(account.id, updated_portfolio)
                st.session_state.portfolio = updated_portfolio
                saved("Portfolio saved")
            else:
                store.delete_portfolio(account.id)
                st.session_state.portfolio = None
                saved("Portfolio removed")

st.page_link("app_pages/assistant.py", label="Back to Finnie", icon=":material/arrow_back:")
