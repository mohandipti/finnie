"""Portfolio tab: a dashboard for the sample portfolio (static data for now)."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.sample_data import load_sample_portfolio
from src.web_app.components.charts import series_colors, style

ASSET_TYPE_LABELS = {"stock": "Stocks", "etf": "Stock funds (ETFs)", "bond": "Bonds", "cash": "Cash",
                     "mutual_fund": "Mutual funds", "other": "Other"}
# Simple risk weights by asset type, averaged by portfolio weight.
RISK_WEIGHTS = {"cash": 1, "bond": 2, "etf": 3, "mutual_fund": 3, "stock": 4, "other": 5}


def compute_metrics(holdings: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.DataFrame(holdings)
    df["value"] = df["quantity"] * df["price"]
    df["cost"] = df["quantity"] * df["cost_basis"]
    df["gain"] = df["value"] - df["cost"]
    total = df["value"].sum()
    df["weight"] = df["value"] / total

    risk_score = float((df["weight"] * df["asset_type"].map(RISK_WEIGHTS)).sum())
    metrics = {
        "total_value": total,
        "gain": df["gain"].sum(),
        "gain_pct": df["gain"].sum() / df["cost"].sum(),
        "expense_ratio": float((df["weight"] * df["expense_ratio"]).sum()),
        # 100 × (1 − Herfindahl index of holding weights): 0 = all in one holding.
        "diversification": 100 * (1 - float((df["weight"] ** 2).sum())),
        "risk_score": risk_score,
        "risk_level": "Low" if risk_score < 2.5 else "Medium" if risk_score < 3.5 else "High",
    }
    return df, metrics


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
    fig.update_xaxes(title="% of portfolio", ticksuffix="%")
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


def render() -> None:
    data = load_sample_portfolio()
    st.info(f"Sample portfolio with fixed prices as of {data['as_of']} — not live data. "
            "You'll be able to enter your own holdings after sign-up.", icon="ℹ️")
    df, m = compute_metrics(data["holdings"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total value", f"${m['total_value']:,.0f}")
    c2.metric("Gain / loss", f"${m['gain']:,.0f}", f"{m['gain_pct']:+.1%}")
    c3.metric("Diversification", f"{m['diversification']:.0f} / 100",
              help="100 × (1 − sum of squared holding weights). Higher means money is spread across more holdings.")
    c4.metric("Risk level", m["risk_level"],
              help=f"Average of asset-type risk weights (cash 1 … stocks 4), weighted by value: {m['risk_score']:.1f}")
    c5.metric("Avg. fund fees", f"{m['expense_ratio']:.2%}",
              help="Value-weighted expense ratio; individual stocks and cash count as 0%.")

    left, right = st.columns(2)
    left.plotly_chart(_allocation_by_holding(df), use_container_width=True, theme="streamlit")
    right.plotly_chart(_allocation_by_type(df), use_container_width=True, theme="streamlit")

    st.subheader("Holdings")
    table = df.assign(
        asset_type=df["asset_type"].map(ASSET_TYPE_LABELS), weight=df["weight"] * 100,
        expense_ratio=df["expense_ratio"] * 100,
    )[["ticker", "name", "asset_type", "quantity", "price", "value", "gain", "weight", "expense_ratio"]]
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "ticker": "Ticker", "name": "Name", "asset_type": "Type", "quantity": "Quantity",
            "price": st.column_config.NumberColumn("Price", format="dollar"),
            "value": st.column_config.NumberColumn("Value", format="dollar"),
            "gain": st.column_config.NumberColumn("Gain / loss", format="dollar"),
            "weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "expense_ratio": st.column_config.NumberColumn("Expense ratio", format="%.2f%%"),
        },
    )
