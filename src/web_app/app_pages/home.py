"""Home page: an introduction to Finnie with links to sign up, log in or try the assistant."""

import streamlit as st

from src.core.disclaimers import DISCLAIMER
from src.web_app.auth import current_account, log_out

FEATURES = [
    (":material/school:", "Learns at your level",
     ("Tell Finnie whether you're a beginner, intermediate or advanced investor. The same question gets a "
     "plain-language explanation with everyday examples, or a precise one with the technical details.")),
    (":material/menu_book:", "Grounded in trusted sources",
     ("Answers draw on a curated library of finance articles from Investor.gov (U.S. SEC), the Federal Reserve "
     "and the IRS, and every answer lists the articles it used.")),
    (":material/verified_user:", "Education, not advice",
     ("Finnie explains how things work so you can make your own decisions. It doesn't tell you what to buy or "
     "sell, and every answer carries a clear disclaimer.")),
]

AGENTS = [
    ("Finance Q&A", "Explains financial concepts, from compound interest to ETFs.", True),
    ("Portfolio analysis", "Reviews your holdings at current prices: allocation, diversification, fees and risk.",
     True),
    ("Market analysis", "Explains current prices, trends and volatility for US stocks and ETFs.", True),
    ("Goal planning", "Helps you plan for goals like a house or retirement, keeping your risk appetite in mind.",
     True),
    ("News synthesizer", "Summarizes recent headlines about a company or topic and explains why they matter.",
     True),
    ("Tax education", "Explains tax concepts and account types such as IRAs and 401(k)s.", True),
]

# Pin the account links into the top-right corner of Streamlit's header bar, where the
# Deploy button and menu used to be (both hidden in .streamlit/config.toml).
st.html("""
<style>
.st-key-header-links {
    position: fixed;
    top: 0;
    right: 1rem;
    height: 3.75rem;
    width: auto;
    z-index: 999991;
    align-items: center;
}
</style>
""")

with st.container(key="header-links", horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
    account = current_account()
    if account:
        st.page_link("app_pages/assistant.py", label=f"Open Finnie as {account.name}", icon=":material/forum:")
        st.page_link("app_pages/account.py", label="My account", icon=":material/account_circle:")
        st.button("Log out", on_click=log_out, icon=":material/logout:", type="tertiary")
    else:
        st.page_link("app_pages/signup.py", label="Sign up", icon=":material/person_add:")
        st.page_link("app_pages/login.py", label="Log in", icon=":material/login:")

with st.container(horizontal_alignment="center"):
    st.title("Meet Finnie", anchor=False, text_alignment="center")
    st.markdown(
        "Investing can feel intimidating: unfamiliar terms, endless choices and the fear of getting it wrong. "
        "Finnie is an AI finance assistant that explains money and investing in clear language, tailored to "
        "what you already know.",
        text_alignment="center",
        width=720,
    )

st.space("large")
st.header("Why Finnie", anchor=False)
for column, (icon, title, text) in zip(st.columns(len(FEATURES)), FEATURES, strict=True):
    with column.container(border=True, height="stretch"):
        st.subheader(f"{icon} {title}", anchor=False)
        st.markdown(text)

st.space("large")
st.header("A team of specialists", anchor=False)
st.markdown("Finnie is a multi-agent system: each agent focuses on one area of personal finance.")
columns = st.columns(3)
for i, (name, text, available) in enumerate(AGENTS):
    with columns[i % 3].container(border=True, height="stretch"):
        st.markdown(f"**{name}**")
        if available:
            st.badge("Available", icon=":material/check:", color="green")
        else:
            st.badge("Coming soon", icon=":material/schedule:", color="gray")
        st.markdown(text)

st.space("large")
st.divider()
st.caption(DISCLAIMER)
st.caption(":material/flag: Finnie currently supports the US stock market only: US-listed stocks, ETFs and funds, "
           "priced in US dollars.")
