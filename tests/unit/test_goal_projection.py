import pytest

from src.core.goal_projection import format_projection, future_value, months_to_target, project_goal


def house(**overrides):
    args = {"name": "House down payment", "target_amount": 60_000, "horizon_years": 5, "risk_tolerance": "moderate",
            "current_savings": 15_000, "monthly_contribution": 600}
    return project_goal(**{**args, **overrides})


def test_projection_with_a_low_and_high_case():
    p = house()
    assert (p.annual_return, p.low_return, p.high_return) == (0.06, 0.04, 0.08)
    assert round(p.expected_value) == 62_095
    assert (round(p.low_value), round(p.high_value)) == (58_094, 66_434)
    assert p.on_track and round(p.gap) == 2_095
    assert round(p.required_monthly) == 570
    assert p.months_to_target == pytest.approx(57.68, abs=0.01)
    assert p.total_contributed == 51_000


def test_portfolio_amount_is_added_to_current_savings():
    p = house(from_portfolio=10_000)
    assert p.current_savings == 25_000 and p.from_portfolio == 10_000
    assert round(p.expected_value) == round(future_value(25_000, 600, 0.06, 60))
    assert "plus $10,000 the user chose to count from their portfolio" in format_projection(p)


def test_already_reached_and_unreachable_goals():
    reached = house(current_savings=70_000)
    assert reached.required_monthly == 0 and reached.months_to_target == 0
    assert months_to_target(60_000, 0, 0, 0.06) is None


def test_zero_return_is_simple_saving():
    p = house(risk_tolerance="conservative", current_savings=0, monthly_contribution=1_000)
    assert p.low_return == 0.02
    assert future_value(0, 1_000, 0.0, 60) == 60_000


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        house(target_amount=0)
    with pytest.raises(ValueError):
        house(monthly_contribution=-1)
