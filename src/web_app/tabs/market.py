"""Market tab: an overview of indexes and a price trend chart (static sample data for now)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.sample_data import load_sample_market
from src.web_app.components.charts import series_colors, style

PERIODS = {"3 months": 63, "6 months": 126, "1 year": 252}
HISTORY_DAYS = 252 + 200  # enough history for a 200-day average over the longest period


@st.cache_data
def sample_prices(ticker: str, start_price: float, drift: float, volatility: float, as_of: str) -> pd.DataFrame:
    """A deterministic illustrative price path (seeded random walk), with 50/200-day moving averages."""
    rng = np.random.default_rng(sum(map(ord, ticker)))
    returns = rng.normal(drift, volatility, HISTORY_DAYS)
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=HISTORY_DAYS)
    df = pd.DataFrame({"price": start_price * np.exp(np.cumsum(returns))}, index=dates)
    df["sma50"] = df["price"].rolling(50).mean()
    df["sma200"] = df["price"].rolling(200).mean()
    return df


def _price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    colors = series_colors()
    fig = go.Figure()
    for column, label, color, width in [("price", "Price", colors[0], 2), ("sma50", "50-day average", colors[1], 2),
                                        ("sma200", "200-day average", colors[2], 2)]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[column], name=label, mode="lines", line={"color": color, "width": width},
            hovertemplate=f"{label}: $%{{y:,.2f}}<extra></extra>",
        ))
    last = df.iloc[-1]
    fig.add_annotation(x=df.index[-1], y=last["price"], text=f"${last['price']:,.2f}", showarrow=False,
                       xanchor="left", xshift=6)
    fig.update_layout(title=f"{ticker} price and moving averages", hovermode="x unified")
    fig.update_yaxes(tickprefix="$")
    return style(fig, height=400)


def render() -> None:
    data = load_sample_market()
    st.info(f"Illustrative sample data as of {data['as_of']}: index levels and price paths are not real. "
            "Live quotes will come from the market data provider.", icon="ℹ️")

    columns = st.columns(len(data["indexes"]))
    for column, index in zip(columns, data["indexes"]):
        column.metric(index["name"], f"{index['value']:,.0f}", f"{index['change_pct']:+.2f}%")

    tickers = data["tickers"]
    ticker_col, period_col = st.columns([2, 1])
    ticker = ticker_col.selectbox("Ticker", list(tickers), format_func=lambda t: f"{t} · {tickers[t]['name']}")
    period = period_col.segmented_control("Period", list(PERIODS), default="1 year") or "1 year"
    days = PERIODS[period]

    params = tickers[ticker]
    df = sample_prices(ticker, params["start_price"], params["daily_drift"], params["daily_volatility"], data["as_of"])
    window = df.tail(days)
    st.plotly_chart(_price_chart(window, ticker), use_container_width=True, theme="streamlit")

    change = window["price"].iloc[-1] / window["price"].iloc[0] - 1
    volatility = window["price"].pct_change().std() * np.sqrt(252)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Change over {period}", f"{change:+.1%}")
    c2.metric("Annualized volatility", f"{volatility:.1%}", help="Standard deviation of daily returns × √252.")
    above = window["price"].iloc[-1] > window["sma200"].iloc[-1]
    c3.metric("vs 200-day average", "Above" if above else "Below",
              help="Price above its 200-day average is often described as a long-term uptrend; it is not a prediction.")

    with st.expander("Show data table"):
        st.dataframe(window.round(2).rename(columns={"price": "Price", "sma50": "50-day avg", "sma200": "200-day avg"}),
                     use_container_width=True)
