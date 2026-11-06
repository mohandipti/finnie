"""Portfolio tab: a dashboard for the signed-in user's saved portfolio."""

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.models import Portfolio, Quote
from src.core.portfolio_metrics import compute_metrics, price_holdings
from src.data.market_data import MarketDataService
from src.web_app.components.charts import series_colors, style

ASSET_TYPE_LABELS = {"stock": "Stocks", "etf": "ETFs", "bond": "Bonds", "cash": "Cash",
                     "mutual_fund": "Mutual funds", "other": "Other"}
# Fresh and cached quotes are both market prices; the caption above the table says how old they are.
PRICE_SOURCE_LABELS = {"alpha_vantage": "Market", "cache": "Market", "cost_basis": "Price paid", "cash": "Cash"}


@st.cache_resource(show_spinner=False)
def get_market_data() -> MarketDataService:
    return MarketDataService.from_settings()


def _allocation_by_holding(df: pd.DataFrame) -> go.Figure:
    data = df.sort_values("weight")
    fig = go.Figure(go.Bar(
        x=data["weight"] * 100, y=data["ticker"], orientation="h",
        marker={"color": series_colors()[0], "cornerradius": 4},
        text=[f"{w:.0%}" for w in data["weight"]], textposition="outside",
        customdata=data[["name", "value"]],
        hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>$%{customdata[1]:,.0f} (%{x:.1f}%)<extra></extra>",
    ))
    fig.update_layout(title="Allocation by holding", bargap=0.35)
    # Leave room past the longest bar for its label.
    fig.update_xaxes(title="% of portfolio", ticksuffix="%", range=[0, data["weight"].max() * 100 * 1.15])
    return style(fig, height=320, legend=False)


def _allocation_by_type(df: pd.DataFrame) -> go.Figure:
    by_type = df.groupby("asset_type", sort=False)["value"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Pie(
        labels=[ASSET_TYPE_LABELS.get(t, t) for t in by_type.index], values=by_type.values,
        hole=0.55, sort=False, direction="clockwise",
        marker={"colors": series_colors()[: len(by_type)], "line": {"width": 2}},
        texttemplate="%{label}<br>%{percent:.0%}", textposition="outside",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Mix by asset type")
    return style(fig, height=320, legend=False)


def _dashboard(df: pd.DataFrame, m: dict[str, Any]) -> None:
    metrics = [
        ("Total value", f"${m['total_value']:,.0f}", None, None),
        ("Gain / loss", "—" if m["gain"] is None else f"${m['gain']:,.0f}",
         None if m["gain_pct"] is None else f"{m['gain_pct']:+.1%}",
         "Compared with the price paid; holdings without a price paid are left out."),
        ("Diversification", f"{m['diversification']:.0f} / 100", None,
         "100 × (1 − sum of squared holding weights). Higher means money is spread across more holdings."),
        ("Risk level", m["risk_level"], None,
         f"Average of asset-type risk weights (cash 1 … stocks 4), weighted by value: {m['risk_score']:.1f}"),
    ]
    if m["expense_ratio"] is not None:
        metrics.append(("Avg. fund fees", f"{m['expense_ratio']:.2%}", None,
                        "Value-weighted expense ratio; individual stocks and cash count as 0%."))
    for column, (label, value, delta, help_text) in zip(st.columns(len(metrics)), metrics, strict=True):
        column.metric(label, value, delta, help=help_text)

    left, right = st.columns(2)
    left.plotly_chart(_allocation_by_holding(df), width="stretch", theme="streamlit")
    right.plotly_chart(_allocation_by_type(df), width="stretch", theme="streamlit")

    st.subheader("Holdings", anchor=False)
    table = df.assign(asset_type=df["asset_type"].map(ASSET_TYPE_LABELS), weight=df["weight"] * 100,
                      expense_ratio=pd.to_numeric(df["expense_ratio"]) * 100)
    # Saved holdings have no names, so the column would only repeat the ticker.
    columns = ["ticker", "name"] if (table["name"] != table["ticker"]).any() else ["ticker"]
    columns += ["asset_type", "quantity", "price"]
    if table["change_pct"].notna().any():
        columns.append("change_pct")
    columns += ["value", "gain", "weight"]
    if "price_source" in table:
        table["price_source"] = table["price_source"].map(PRICE_SOURCE_LABELS)
        columns.append("price_source")
    if table["expense_ratio"].notna().any():
        columns.append("expense_ratio")
    st.dataframe(
        table[columns], hide_index=True, width="stretch", placeholder="—",
        alt="Portfolio holdings with prices and values",
        column_config={
            "ticker": "Ticker", "name": "Name", "asset_type": "Type", "quantity": "Quantity",
            "price": st.column_config.NumberColumn("Price", format="dollar"),
            "change_pct": st.column_config.NumberColumn("Today", format="%+.2f%%"),
            "value": st.column_config.NumberColumn("Value", format="dollar"),
            "gain": st.column_config.NumberColumn("Gain / loss", format="dollar"),
            "weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "price_source": "Price from",
            "expense_ratio": st.column_config.NumberColumn("Expense ratio", format="%.2f%%"),
        },
    )


def _price_note(quotes: dict[str, Quote], rows: list[dict[str, Any]]) -> str:
    parts = []
    if quotes:
        oldest: datetime = min(q.as_of for q in quotes.values())
        parts.append(f"Prices from Alpha Vantage (delayed), oldest fetched {oldest:%b %d, %H:%M} UTC.")
    if any(row["price_source"] == "cost_basis" for row in rows):
        parts.append("Holdings without a current quote are valued at the price paid.")
    return " ".join(parts)


def _render_user_portfolio(portfolio: Portfolio) -> None:
    tickers = [h.ticker for h in portfolio.holdings if h.asset_type != "cash"]
    with st.spinner("Getting current prices..."):
        quotes = get_market_data().get_quotes(tickers)
    rows, unpriced = price_holdings(portfolio, quotes)
    if note := _price_note(quotes, rows):
        st.caption(note)
    if unpriced:
        st.warning(f"No price is available for {', '.join(unpriced)}, so they're left out of the totals. "
                   "Check the ticker, or add the price paid.", icon=":material/warning:")
    if rows:
        _dashboard(*compute_metrics(rows))


def render(portfolio: Portfolio | None) -> None:
    if portfolio is not None:
        _render_user_portfolio(portfolio)
        return
    st.info("You haven't added a portfolio yet. Add your holdings and cash on your account page to see "
            "their value, allocation and risk here.", icon=":material/info:")
    st.page_link("app_pages/account.py", label="Add your portfolio", icon=":material/account_circle:")
