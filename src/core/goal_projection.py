"""Savings goal projections used by the Goal Planning Agent.

Numbers are computed here in code; the agent's LLM only explains them. Growth is
compounded monthly at an assumed annual return for the goal's risk tolerance, with
a low and high case either side of it.
"""

import math

from pydantic import BaseModel

from src.core.models import RiskTolerance

# Assumed average annual returns, not forecasts.
ASSUMED_RETURNS: dict[RiskTolerance, float] = {"conservative": 0.04, "moderate": 0.06, "aggressive": 0.08}
RETURN_SPREAD = 0.02  # low and high cases: the assumed return minus and plus this
INFLATION = 0.025
MAX_MONTHS = 100 * 12  # beyond this the goal is treated as not reachable


class GoalProjection(BaseModel):
    name: str
    target_amount: float
    current_savings: float  # includes any amount counted from the portfolio
    from_portfolio: float
    monthly_contribution: float
    horizon_years: float
    risk_tolerance: RiskTolerance
    annual_return: float
    low_return: float
    high_return: float
    expected_value: float
    low_value: float
    high_value: float
    gap: float  # expected value minus target; negative means short
    on_track: bool
    required_monthly: float  # to reach the target at the assumed return
    required_monthly_low: float  # to reach it even at the low return
    months_to_target: float | None  # at the current contribution; None if not reachable
    total_contributed: float
    target_in_todays_dollars: float


def future_value(present: float, monthly: float, annual_rate: float, months: float) -> float:
    i = annual_rate / 12
    if i == 0:
        return present + monthly * months
    growth = (1 + i) ** months
    return present * growth + monthly * (growth - 1) / i


def required_monthly(target: float, present: float, annual_rate: float, months: float) -> float:
    """Monthly contribution needed to reach `target`; 0 if current savings already get there."""
    shortfall = target - future_value(present, 0, annual_rate, months)
    if shortfall <= 0:
        return 0.0
    i = annual_rate / 12
    if i == 0:
        return shortfall / months
    return shortfall * i / ((1 + i) ** months - 1)


def months_to_target(target: float, present: float, monthly: float, annual_rate: float) -> float | None:
    """Months until savings reach `target` at the current contribution, or None if never (within 100 years)."""
    if present >= target:
        return 0.0
    i = annual_rate / 12
    if i == 0:
        months = (target - present) / monthly if monthly > 0 else math.inf
    elif present == 0 and monthly == 0:
        months = math.inf
    else:
        months = math.log((target * i + monthly) / (present * i + monthly)) / math.log(1 + i)
    return months if months <= MAX_MONTHS else None


def project_goal(
    name: str,
    target_amount: float,
    horizon_years: float,
    risk_tolerance: RiskTolerance = "moderate",
    current_savings: float = 0.0,
    monthly_contribution: float = 0.0,
    from_portfolio: float = 0.0,
) -> GoalProjection:
    if target_amount <= 0 or horizon_years <= 0:
        raise ValueError("target_amount and horizon_years must be positive")
    if min(current_savings, monthly_contribution, from_portfolio) < 0:
        raise ValueError("savings, contributions and portfolio amounts can't be negative")

    rate = ASSUMED_RETURNS[risk_tolerance]
    low_rate, high_rate = round(max(rate - RETURN_SPREAD, 0.0), 4), round(rate + RETURN_SPREAD, 4)
    months = horizon_years * 12
    present = current_savings + from_portfolio
    expected = future_value(present, monthly_contribution, rate, months)
    return GoalProjection(
        name=name,
        target_amount=target_amount,
        current_savings=present,
        from_portfolio=from_portfolio,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        risk_tolerance=risk_tolerance,
        annual_return=rate,
        low_return=low_rate,
        high_return=high_rate,
        expected_value=expected,
        low_value=future_value(present, monthly_contribution, low_rate, months),
        high_value=future_value(present, monthly_contribution, high_rate, months),
        gap=expected - target_amount,
        on_track=expected >= target_amount,
        required_monthly=required_monthly(target_amount, present, rate, months),
        required_monthly_low=required_monthly(target_amount, present, low_rate, months),
        months_to_target=months_to_target(target_amount, present, monthly_contribution, rate),
        total_contributed=present + monthly_contribution * months,
        target_in_todays_dollars=target_amount / (1 + INFLATION) ** horizon_years,
    )


def _money(value: float) -> str:
    return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def _duration(months: float) -> str:
    years, rest = divmod(math.ceil(months), 12)
    parts = [f"{years} year{'s' * (years != 1)}"] if years else []
    if rest:
        parts.append(f"{rest} month{'s' * (rest != 1)}")
    return " ".join(parts) or "now"


def format_projection(p: GoalProjection) -> str:
    """The calculated figures as plain text for the prompt."""
    savings = f"Current savings counted: {_money(p.current_savings)}"
    if p.from_portfolio:
        savings += (f" ({_money(p.current_savings - p.from_portfolio)} saved for the goal plus "
                    f"{_money(p.from_portfolio)} the user chose to count from their portfolio)")
    if p.months_to_target is None:
        timing = "Not reachable within 100 years at the current contribution"
    else:
        timing = f"Time to reach the target at the current contribution: about {_duration(p.months_to_target)}"
    return "\n".join([
        f"Goal: {p.name}",
        f"Target: {_money(p.target_amount)} in {p.horizon_years:g} years ({p.risk_tolerance} risk)",
        savings,
        f"Monthly contribution: {_money(p.monthly_contribution)}",
        f"Assumed annual return: {p.annual_return:.0%} (low case {p.low_return:.0%}, high case {p.high_return:.0%})",
        f"Projected value: {_money(p.expected_value)} (low {_money(p.low_value)}, high {_money(p.high_value)})",
        f"{'Ahead of' if p.gap >= 0 else 'Short of'} target by {_money(abs(p.gap))} at the assumed return",
        (f"Monthly contribution needed: {_money(p.required_monthly)} at {p.annual_return:.0%}, "
         f"{_money(p.required_monthly_low)} at {p.low_return:.0%}"),
        timing,
        f"Total the user puts in: {_money(p.total_contributed)}; the rest of the projected value is growth",
        f"The target in today's dollars, at {INFLATION:.1%} inflation: about {_money(p.target_in_todays_dollars)}",
    ])
