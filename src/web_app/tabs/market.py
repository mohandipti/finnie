"""Market tab: index snapshot and a ticker's price trend, from live or cached Alpha Vantage data.

Prices are fetched only while the tab is open, and share the market data cache with the
Portfolio tab and the Market agent, so a ticker already looked up anywhere costs no request.
"""

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.models import Portfolio, PriceHistory, Quote
from src.data.market_data import TICKER_PATTERN, WEEKS_PER_YEAR
from src.web_app.components.charts import series_colors, style
from src.web_app.components.signup_forms import CASH_TICKER
from src.web_app.tabs.portfolio import get_market_data

# Index values aren't on Alpha Vantage's free tier, so each index is shown through an ETF that tracks it.
INDEX_ETFS = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "DIA": "Dow Jones"}
POPULAR_TICKERS = ["SPY", "VTI", "QQQ", "AAPL", "MSFT"]
PERIODS = {"3 months": 13, "6 months": 26, "1 year": 52, "5 years": 260}  # in weeks
SOURCE_LABELS = {"alpha_vantage": "live", "cache": "cached", "mock": "sample"}


def history_frame(history: PriceHistory) -> pd.DataFrame:
    """Weekly closes with 10- and 40-week moving averages (≈ 50- and 200-day), indexed by date."""
    df = pd.DataFrame([{"date": p.date, "price": p.close} for p in history.points]).set_index("date")
    df.index = pd.to_datetime(df.index)
    df["sma10"] = df["price"].rolling(10).mean()
    df["sma40"] = df["price"].rolling(40).mean()
    return df


def period_stats(df: pd.DataFrame, weeks: int) -> dict[str, float | bool | None]:
    """Change, annualized volatility and position vs the 40-week average over the last `weeks` weeks."""
    window = df.tail(weeks + 1)  # weeks + 1 closes give `weeks` weekly changes
    returns = window["price"].pct_change().dropna()
    last = window.iloc[-1]
    return {
        "change": last["price"] / window["price"].iloc[0] - 1 if len(window) > 1 else None,
        "volatility": returns.std() * math.sqrt(WEEKS_PER_YEAR) if len(returns) >= 2 else None,
        "above_sma40": None if pd.isna(last["sma40"]) else bool(last["price"] > last["sma40"]),
    }


def ticker_options(portfolio: Portfolio | None) -> list[str]:
    """The user's own holdings first, then a few common tickers."""
    held = [h.ticker for h in portfolio.holdings if h.ticker != CASH_TICKER] if portfolio else []
    return list(dict.fromkeys(held + POPULAR_TICKERS))


def _retrieved(as_of: datetime, source: str) -> str:
    return f"retrieved {as_of:%b %d, %H:%M} UTC ({SOURCE_LABELS.get(source, source)})"


def _price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    colors = series_colors()
    fig = go.Figure()
    for column, label, color in [("price", "Weekly close", colors[0]),
                                 ("sma10", "10-week average (≈ 50-day)", colors[1]),
                                 ("sma40", "40-week average (≈ 200-day)", colors[2])]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[column], name=label, mode="lines", line={"color": color, "width": 2},
            hovertemplate=f"{label}: $%{{y:,.2f}}<extra></extra>",
        ))
    last = df.iloc[-1]
    fig.add_annotation(x=df.index[-1], y=last["price"], text=f"${last['price']:,.2f}", showarrow=False,
                       xanchor="left", xshift=6)
    fig.update_layout(title=f"{ticker} price and moving averages", hovermode="x unified")
    fig.update_yaxes(tickprefix="$")
    return style(fig, height=400)


def _index_cards(quotes: dict[str, Quote]) -> None:
    with st.container(horizontal=True):
        for etf, index_name in INDEX_ETFS.items():
            quote = quotes.get(etf)
            st.metric(f"{index_name} ({etf})", f"${quote.price:,.2f}" if quote else "—",
                      f"{quote.change_pct:+.2f}%" if quote else None, border=True,
                      help=f"{etf} is an ETF that tracks the {index_name}; its price moves with the index.")
    if quotes:
        oldest = min(quotes.values(), key=lambda q: q.as_of)
        st.caption(f"Latest quotes from Alpha Vantage, {_retrieved(oldest.as_of, oldest.source)}. "
                   "Change is since the previous close.")
    else:
        st.caption("Quotes are unavailable right now: the market data provider can't be reached, "
                   "or today's request limit has been reached.")


def render(portfolio: Portfolio | None = None) -> None:
    data = get_market_data()
    with st.spinner("Fetching market data..."):
        quotes = data.get_quotes(list(INDEX_ETFS))
    _index_cards(quotes)

    ticker_col, period_col = st.columns([2, 1], vertical_alignment="bottom")
    choice = ticker_col.selectbox("Ticker", ticker_options(portfolio), accept_new_options=True,
                                  key="market_ticker", help="Pick one, or type any US-listed ticker.")
    period = period_col.segmented_control("Period", list(PERIODS), default="1 year", key="market_period") or "1 year"
    ticker = (choice or "").strip().upper()
    if not TICKER_PATTERN.match(ticker):
        st.warning(f"“{choice}” doesn't look like a US ticker symbol (for example AAPL or BRK.B).",
                   icon=":material/search_off:")
        return

    with st.spinner(f"Fetching {ticker} price history..."):
        history = data.get_history(ticker)
    if history is None:
        st.warning(f"No price history for {ticker}: it may not be a US-listed ticker, or the market data "
                   "provider can't be reached right now.", icon=":material/cloud_off:")
        return

    df = history_frame(history)
    weeks = PERIODS[period]
    st.plotly_chart(_price_chart(df.tail(weeks + 1), ticker), width="stretch", theme="streamlit")
    st.caption(f"Weekly closes adjusted for splits and dividends, up to {df.index[-1]:%b %d, %Y}; "
               f"{_retrieved(history.as_of, history.source)}.")

    stats = period_stats(df, weeks)
    with st.container(horizontal=True):
        st.metric(f"Change over {period}", "—" if stats["change"] is None else f"{stats['change']:+.1%}", border=True)
        st.metric("Annualized volatility", "—" if stats["volatility"] is None else f"{stats['volatility']:.1%}",
                  border=True, help="Standard deviation of weekly returns over the period × √52.")
        above = stats["above_sma40"]
        st.metric("vs 40-week average", "—" if above is None else ("Above" if above else "Below"), border=True,
                  help="The 40-week average is close to the 200-day average. A price above it is often described "
                       "as a long-term uptrend; it is not a prediction.")

    with st.expander("Show data table"):
        table = df.tail(weeks + 1).sort_index(ascending=False).round(2)
        st.dataframe(table.rename(columns={"price": "Close", "sma10": "10-week avg", "sma40": "40-week avg"}),
                     column_config={"_index": st.column_config.DateColumn("Week")}, placeholder="—")
