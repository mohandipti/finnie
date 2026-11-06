from streamlit.testing.v1 import AppTest


def chat_with_open_question():
    import streamlit as st

    from src.web_app.tabs import chat
    from src.workflow.graph import FinnieReply, StreamEvent

    class FakeGraph:
        def resume_stream(self, session_id, answer):
            st.session_state.resumed_with = answer
            reply = FinnieReply(content=f"Projected with ${answer['amount']:,.0f}", agents=["goal_planning"])
            yield StreamEvent("final", reply=reply)

    st.session_state.setdefault("session_id", "s1")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("goal_question", {
        "goal": "House down payment", "goal_savings": 15_000.0, "portfolio_value": 39_692.0,
        "question": "Your portfolio is worth about **$39,692**. How much should count toward this goal?"})
    chat.render(FakeGraph(), None, None, None)


def test_open_question_is_answered_from_the_form():
    at = AppTest.from_function(chat_with_open_question).run()
    assert not at.exception
    assert any("$39,692" in m.value for m in at.markdown)

    at.number_input[0].set_value(10_000)
    next(b for b in at.button if b.label == "Count it").click().run()

    assert not at.exception
    assert at.session_state.resumed_with == {"amount": 10_000}
    assert "goal_question" not in at.session_state
    assert [m["content"] for m in at.session_state.messages][1:] == [
        "Count $10,000 of my portfolio toward House down payment.", "Projected with $10,000"]


def test_skip_counts_nothing():
    at = AppTest.from_function(chat_with_open_question).run()
    next(b for b in at.button if b.label == "Skip").click().run()
    assert at.session_state.resumed_with == {"amount": 0.0}
