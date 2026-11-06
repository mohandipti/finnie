"""Convert between saved goals and holdings and the editable tables on the sign-up and My account pages."""

from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.core.models import FinancialGoal, Holding, Portfolio, RiskTolerance

Model = TypeVar("Model", bound=BaseModel)

GOAL_COLUMNS = ["name", "target_amount", "current_savings", "monthly_contribution", "horizon_years",
                "risk_tolerance", "from_portfolio"]
HOLDING_COLUMNS = ["ticker", "quantity", "cost_basis", "asset_type"]
# Cash has its own field on the form; it is stored as a holding like the sample portfolio's.
INVESTMENT_TYPES = ["stock", "etf", "mutual_fund", "bond", "other"]
CASH_TICKER = "CASH"
FIELD_LABELS = {
    "name": "Goal", "target_amount": "Target", "current_savings": "Saved so far",
    "monthly_contribution": "Monthly contribution", "horizon_years": "Years", "risk_tolerance": "Risk",
    "from_portfolio": "From portfolio",
    "ticker": "Ticker", "quantity": "Quantity", "cost_basis": "Price paid", "asset_type": "Type",
}


def empty_goals() -> pd.DataFrame:
    return pd.DataFrame({
        "name": pd.Series(dtype="str"), "target_amount": pd.Series(dtype="float"),
        "current_savings": pd.Series(dtype="float"), "monthly_contribution": pd.Series(dtype="float"),
        "horizon_years": pd.Series(dtype="float"), "risk_tolerance": pd.Series(dtype="str"),
        "from_portfolio": pd.Series(dtype="float"),
    })


def empty_holdings() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": pd.Series(dtype="str"), "quantity": pd.Series(dtype="float"),
        "cost_basis": pd.Series(dtype="float"), "asset_type": pd.Series(dtype="str"),
    })


def goals_frame(goals: list[FinancialGoal]) -> pd.DataFrame:
    """Saved goals as an editable table."""
    if not goals:
        return empty_goals()
    return pd.DataFrame([g.model_dump() for g in goals]).reindex(columns=GOAL_COLUMNS)


def holdings_frame(portfolio: Portfolio | None) -> tuple[pd.DataFrame, float]:
    """A saved portfolio as an editable holdings table (investments only) and the cash amount."""
    if portfolio is None:
        return empty_holdings(), 0.0
    investments = [h.model_dump() for h in portfolio.holdings if h.asset_type != "cash"]
    cash = sum(h.quantity for h in portfolio.holdings if h.asset_type == "cash")
    frame = pd.DataFrame(investments).reindex(columns=HOLDING_COLUMNS) if investments else empty_holdings()
    return frame, cash


def _records(df: pd.DataFrame, columns: list[str]) -> list[tuple[int, dict]]:
    """Non-blank rows as (row number, {column: value}), with blanks as None."""
    rows = []
    for number, (_, row) in enumerate(df.reindex(columns=columns).iterrows(), start=1):
        record = {col: (None if pd.isna(value) or (isinstance(value, str) and not value.strip()) else value)
                  for col, value in row.items()}
        if any(value is not None for value in record.values()):
            rows.append((number, record))
    return rows


def _describe(error: ValidationError, kind: str, number: int) -> str:
    first = error.errors()[0]
    field = FIELD_LABELS.get(str(first["loc"][0]), str(first["loc"][0])) if first["loc"] else ""
    message = first["msg"].removeprefix("Value error, ")
    return f"{kind} row {number}: {field} — {message}" if field else f"{kind} row {number}: {message}"


def _parse(rows: list[tuple[int, dict]], model: type[Model], kind: str) -> tuple[list[Model], list[str]]:
    items, errors = [], []
    for number, record in rows:
        try:
            items.append(model.model_validate({k: v for k, v in record.items() if v is not None}))
        except ValidationError as exc:
            errors.append(_describe(exc, kind, number))
    return items, errors


def parse_goals(df: pd.DataFrame, default_risk: RiskTolerance) -> tuple[list[FinancialGoal], list[str]]:
    """Goals from the goals table; a goal without its own risk choice uses the profile's."""
    rows = [(n, {**r, "risk_tolerance": r["risk_tolerance"] or default_risk}) for n, r in _records(df, GOAL_COLUMNS)]
    return _parse(rows, FinancialGoal, "Goal")


def parse_holdings(df: pd.DataFrame) -> tuple[list[Holding], list[str]]:
    """Investments from the holdings table (cash is entered separately)."""
    holdings, errors = _parse(_records(df, HOLDING_COLUMNS), Holding, "Holding")
    if any(h.ticker == CASH_TICKER or h.asset_type == "cash" for h in holdings):
        errors.append("Holdings: enter cash in the Cash field instead of the table")
    return holdings, errors


def cash_holding(amount: float | None) -> Holding | None:
    """Cash as a holding: one unit per dollar at $1 each, so it values like any other position."""
    if not amount or amount <= 0:
        return None
    return Holding(ticker=CASH_TICKER, quantity=round(amount, 2), cost_basis=1.0, asset_type="cash")
