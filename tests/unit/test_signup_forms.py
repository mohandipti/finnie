import pandas as pd

from src.core.models import FinancialGoal, Holding, Portfolio
from src.web_app.components.signup_forms import (
    cash_holding,
    empty_goals,
    empty_holdings,
    goals_frame,
    holdings_frame,
    parse_goals,
    parse_holdings,
)


def test_empty_tables_parse_to_nothing():
    assert parse_goals(empty_goals(), "moderate") == ([], [])
    assert parse_holdings(empty_holdings()) == ([], [])


def test_goals_default_to_profile_risk_and_skip_blank_rows():
    df = pd.DataFrame([
        {"name": "House", "target_amount": 60000, "horizon_years": 5, "risk_tolerance": None},
        {"name": None, "target_amount": None, "horizon_years": None, "risk_tolerance": None},
        {"name": "Retire", "target_amount": 900000, "horizon_years": 30, "risk_tolerance": "aggressive"},
    ])
    goals, errors = parse_goals(df, "conservative")
    assert errors == []
    assert [(g.name, g.risk_tolerance) for g in goals] == [("House", "conservative"), ("Retire", "aggressive")]
    assert goals[0].current_savings == 0


def test_invalid_goal_reports_row_and_field():
    df = pd.DataFrame([{"name": "House", "target_amount": 0, "horizon_years": 5}])
    goals, errors = parse_goals(df, "moderate")
    assert goals == []
    assert errors[0].startswith("Goal row 1: Target")


def test_holdings_are_normalized_and_validated():
    df = pd.DataFrame([
        {"ticker": " vti ", "quantity": 10, "cost_basis": 220.0, "asset_type": "etf"},
        {"ticker": "AAPL", "quantity": -1},
    ])
    holdings, errors = parse_holdings(df)
    assert [h.ticker for h in holdings] == ["VTI"]
    assert errors == ["Holding row 2: Quantity — Input should be greater than 0"]


def test_cash_belongs_in_the_cash_field():
    df = pd.DataFrame([{"ticker": "cash", "quantity": 500}])
    _, errors = parse_holdings(df)
    assert errors == ["Holdings: enter cash in the Cash field instead of the table"]


def test_cash_holding():
    cash = cash_holding(2500.456)
    assert (cash.ticker, cash.quantity, cash.cost_basis, cash.asset_type) == ("CASH", 2500.46, 1.0, "cash")
    assert cash_holding(0) is None
    assert cash_holding(None) is None


def test_saved_goals_and_holdings_round_trip_through_the_tables():
    goal = FinancialGoal(name="House", target_amount=60000, horizon_years=5, risk_tolerance="moderate")
    assert parse_goals(goals_frame([goal]), "moderate") == ([goal], [])

    vti = Holding(ticker="VTI", quantity=10, cost_basis=300, asset_type="etf")
    frame, cash = holdings_frame(Portfolio(holdings=[vti, cash_holding(250)]))
    assert cash == 250
    assert parse_holdings(frame) == ([vti], [])
    assert holdings_frame(None)[1] == 0
