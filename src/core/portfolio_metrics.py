"""Portfolio calculations shared by the Portfolio tab and the Portfolio Analysis Agent.

Numbers are computed here in code; the agent's LLM only explains them.
"""

from typing import Any

import pandas as pd

from src.core.models import Portfolio, Quote

# Simple risk weights by asset type, averaged by portfolio weight.
RISK_WEIGHTS = {"cash": 1, "bond": 2, "etf": 3, "mutual_fund": 3, "stock": 4, "other": 5}


def compute_metrics(holdings: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-holding values and portfolio metrics. cost_basis and expense_ratio may be missing."""
    df = pd.DataFrame(holdings)
    for column in ("name", "cost_basis", "expense_ratio", "change_pct"):
        if column not in df:
            df[column] = None
    df["name"] = df["name"].fillna(df["ticker"])
    df["asset_type"] = df["asset_type"].fillna("other")
    df["cost_basis"] = pd.to_numeric(df["cost_basis"])
    df["value"] = df["quantity"] * df["price"]
    df["cost"] = df["quantity"] * df["cost_basis"]
    df["gain"] = df["value"] - df["cost"]  # NaN where the price paid is unknown
    total = df["value"].sum()
    df["weight"] = df["value"] / total

    known = df["cost"].notna()
    cost, gain = df.loc[known, "cost"].sum(), df.loc[known, "gain"].sum()
    expense = pd.to_numeric(df["expense_ratio"])
    risk_score = float((df["weight"] * df["asset_type"].map(RISK_WEIGHTS)).sum())
    metrics = {
        "total_value": total,
        "gain": gain if known.any() else None,
        "gain_pct": gain / cost if known.any() and cost > 0 else None,
        "expense_ratio": float((df["weight"] * expense).sum()) if expense.notna().any() else None,
        # 100 × (1 − Herfindahl index of holding weights): 0 = all in one holding.
        "diversification": 100 * (1 - float((df["weight"] ** 2).sum())),
        "risk_score": risk_score,
        "risk_level": "Low" if risk_score < 2.5 else "Medium" if risk_score < 3.5 else "High",
    }
    return df, metrics


def price_holdings(portfolio: Portfolio, quotes: dict[str, Quote]) -> tuple[list[dict[str, Any]], list[str]]:
    """Holdings with a price each: the quote, else the price paid. Returns (priced, tickers with no price)."""
    priced, unpriced = [], []
    for holding in portfolio.holdings:
        row = holding.model_dump()
        quote = quotes.get(holding.ticker)
        if holding.asset_type == "cash":
            row.update(price=1.0, price_source="cash")
        elif quote is not None:
            row.update(price=quote.price, price_source=quote.source, change_pct=quote.change_pct)
        elif holding.cost_basis is not None:
            row.update(price=holding.cost_basis, price_source="cost_basis")
        else:
            unpriced.append(holding.ticker)
            continue
        priced.append(row)
    return priced, unpriced
