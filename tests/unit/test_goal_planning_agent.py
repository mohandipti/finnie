from datetime import UTC, datetime

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents.goal_planning_agent import GoalPlanningAgent, goals_in_question
from src.core.models import FinancialGoal, Holding, Portfolio, Quote, UserProfile
from src.workflow.state import FinnieState

NOW = datetime(2026, 11, 5, 14, 0, tzinfo=UTC)
HOUSE = FinancialGoal(name="House down payment", target_amount=60_000, current_savings=15_000,
                      monthly_contribution=600, horizon_years=5, risk_tolerance="moderate")
CAR = FinancialGoal(name="New car", target_amount=20_000, horizon_years=3, risk_tolerance="conservative")
PORTFOLIO = Portfolio(holdings=[Holding(ticker="VTI", quantity=100, asset_type="etf"),
                                Holding(ticker="CASH", quantity=2_000, asset_type="cash")])


class FakeMarketData:
    def get_quotes(self, tickers):
        return {t: Quote(ticker=t, price=300.0, change=0, change_pct=0, as_of=NOW, source="cache") for t in tickers}


class FakeLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def agent(*replies):
    llm = FakeLLM(messages=iter(replies or [AIMessage(content="answer")]))
    return GoalPlanningAgent(llm=llm, market_data=FakeMarketData())


def state(query, goals=(HOUSE,), portfolio=PORTFOLIO, answers=None):
    return {"session_id": "s1", "user_query": query, "messages": [], "portfolio": portfolio,
            "user_profile": UserProfile(session_id="s1", risk_tolerance="moderate", goals=list(goals)),
            "goal_portfolio_answers": answers or {}}


def test_picks_goals_by_name_or_wording():
    profile = UserProfile(session_id="s1", goals=[HOUSE, CAR])
    assert goals_in_question(profile, "Am I on track for the house?") == [HOUSE]
    assert goals_in_question(profile, "How are my goals doing?") == [HOUSE, CAR]
    assert goals_in_question(profile, "Am I on track?") == []  # two goals: ambiguous
    assert goals_in_question(UserProfile(session_id="s1", goals=[CAR]), "Am I on track?") == [CAR]


def test_asks_about_the_portfolio_only_when_needed():
    goals = agent()
    question = goals.question_for_user(state("Am I on track for my house down payment?"))
    assert question["goal"] == "House down payment" and question["portfolio_value"] == 32_000

    assert goals.question_for_user(state("Am I on track for my house?", portfolio=None)) is None
    assert goals.question_for_user(state("Am I on track for my house?", answers={"House down payment": 0})) is None
    saved = HOUSE.model_copy(update={"from_portfolio": 5_000})
    assert goals.question_for_user(state("Am I on track for my house?", goals=[saved])) is None


def test_pauses_then_uses_the_users_amount():
    goals = agent(AIMessage(content="You're on track."))
    graph = StateGraph(FinnieState)
    graph.add_node("goal_planning", goals.as_node())
    graph.add_edge(START, "goal_planning")
    graph.add_edge("goal_planning", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "s1"}}

    app.invoke(state("Am I on track for my house down payment?"), config)
    paused = app.get_state(config)
    assert paused.interrupts and paused.interrupts[0].value["goal"] == "House down payment"

    result = app.invoke(Command(resume={"amount": 50_000}), config)  # more than the portfolio: capped
    assert result["goal_portfolio_answers"] == {"House down payment": 32_000}
    projection = result["agent_outputs"][0].data["projections"][0]
    assert projection["from_portfolio"] == 32_000 and projection["current_savings"] == 47_000


def test_without_a_saved_goal_the_llm_can_call_project_goal():
    tool_call = AIMessage(content="", tool_calls=[{"name": "project_goal", "id": "1", "args": {
        "target_amount": 25_000, "horizon_years": 4, "current_savings": 5_000, "monthly_contribution": 300}}])
    goals = agent(tool_call, AIMessage(content="You'd get close."))
    response = goals.run(state("I have $5,000 and save $300 a month. Can I reach $25,000 in 4 years?", goals=[]))

    assert response.content == "You'd get close." and response.error is None
    assert "Projected value: $" in response.data["tool_results"][0]["result"]
