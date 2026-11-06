"""Profile, goal and holding inputs shared by the sign-up and My account pages."""

import pandas as pd
import streamlit as st

from src.web_app.components.signup_forms import INVESTMENT_TYPES

KNOWLEDGE_LEVELS = {
    "beginner": "Beginner: I'm new to investing",
    "intermediate": "Intermediate: I know the basics",
    "advanced": "Advanced: I'm comfortable with the details",
}
RISK_LEVELS = {
    "conservative": "Conservative: I want to protect what I have, even if it grows slowly",
    "moderate": "Moderate: I can accept some ups and downs for better growth",
    "aggressive": "Aggressive: I'm comfortable with big swings for higher long-term growth",
}
INTERESTS = ["Investing basics", "Stocks", "ETFs and index funds", "Bonds", "Retirement", "Taxes",
             "Budgeting and saving", "Buying a home", "Market news"]


def about_you_inputs(level: str = "beginner", risk: str = "moderate",
                     interests: list[str] | None = None) -> tuple[str, str, list[str]]:
    level = st.radio("How much do you know about investing?", list(KNOWLEDGE_LEVELS),
                     index=list(KNOWLEDGE_LEVELS).index(level), format_func=KNOWLEDGE_LEVELS.get)
    risk = st.radio("How do you feel about risk?", list(RISK_LEVELS), index=list(RISK_LEVELS).index(risk),
                    format_func=RISK_LEVELS.get)
    chosen = st.pills("What would you like to learn about?", INTERESTS, selection_mode="multi",
                      default=[i for i in (interests or []) if i in INTERESTS])
    return level, risk, chosen or []


def goals_editor(data: pd.DataFrame, key: str) -> pd.DataFrame:
    return st.data_editor(
        data, key=key, num_rows="dynamic", hide_index=True, width="stretch", alt="Editable table of financial goals",
        column_config={
            "name": st.column_config.TextColumn("Goal", max_chars=60),
            "target_amount": st.column_config.NumberColumn("Target", min_value=0, format="dollar"),
            "current_savings": st.column_config.NumberColumn("Saved so far", min_value=0, format="dollar"),
            "monthly_contribution": st.column_config.NumberColumn("Monthly contribution", min_value=0,
                                                                  format="dollar"),
            "horizon_years": st.column_config.NumberColumn("Years", min_value=0, max_value=60, step=0.5),
            "risk_tolerance": st.column_config.SelectboxColumn("Risk", options=list(RISK_LEVELS)),
            "from_portfolio": st.column_config.NumberColumn(
                "From portfolio", min_value=0, format="dollar",
                help="How much of your portfolio is set aside for this goal. Leave blank and Finnie will ask."),
        },
    )


def holdings_editor(data: pd.DataFrame, key: str) -> pd.DataFrame:
    return st.data_editor(
        data, key=key, num_rows="dynamic", hide_index=True, width="stretch", alt="Editable table of portfolio holdings",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", max_chars=10, help="For example VTI or AAPL"),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0),
            "cost_basis": st.column_config.NumberColumn("Price paid", min_value=0, format="dollar",
                                                        help="Average price paid per share (optional)"),
            "asset_type": st.column_config.SelectboxColumn("Type", options=INVESTMENT_TYPES),
        },
    )


def cash_input(value: float = 0.0, key: str | None = None) -> float:
    return st.number_input("Cash", value=float(value), min_value=0.0, step=100.0, format="%.2f", width=240, key=key,
                           help="Uninvested cash held in your investment account, in US dollars.")
