"""Chat tab: a conversation with Finnie, streamed from the LangGraph workflow."""

import streamlit as st

from src.core.models import Portfolio, UserProfile
from src.data.user_store import UserStoreUnavailable
from src.web_app.auth import current_account, get_user_store
from src.workflow.graph import FinnieGraph

AGENT_LABELS = {
    "finance_qa": "Finance Q&A",
    "portfolio": "Portfolio",
    "market": "Market",
    "goal_planning": "Goal Planning",
    "news": "News",
    "tax": "Tax",
}
EXAMPLE_PROMPTS = [
    "What is an ETF?",
    "How does compound interest work?",
    "Why do bond prices fall when interest rates rise?",
    "What's the difference between a Roth and a traditional IRA?",
]


def _render_reply_extras(agents: list[str], sources: list) -> None:
    if agents:
        st.caption("Answered by: " + " · ".join(AGENT_LABELS.get(a, a) for a in agents))
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, source in enumerate(sources, start=1):
                line = f"[{i}] [{source.title}]({source.url})" if source.url else f"[{i}] {source.title}"
                st.markdown(line + (f" · *{source.category}*" if source.category else ""))


def render(graph: FinnieGraph | None, profile: UserProfile, portfolio: Portfolio | None,
           startup_error: str | None) -> None:
    if startup_error:
        st.error(startup_error)
        return

    messages: list[dict] = st.session_state.messages
    # Inside a tab, chat_input is drawn where it's called rather than pinned to the bottom. The
    # conversation goes in a container created first, so new messages land above the input.
    conversation = st.container()
    query = st.chat_input("Ask Finnie about money and investing") or st.session_state.pop("pending_prompt", None)

    with conversation:
        if not messages and not query:
            st.markdown("Ask me anything about personal finance and investing basics. Try one of these:")
            columns = st.columns(2)
            for i, prompt in enumerate(EXAMPLE_PROMPTS):
                if columns[i % 2].button(prompt, width="stretch"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    _render_reply_extras(message.get("agents", []), message.get("sources", []))

        if query:
            st.session_state.pop("goal_question", None)  # a new message skips an open question
            _answer(graph, query, profile, portfolio, messages)
        elif st.session_state.get("goal_question"):
            _ask_goal_question(graph, st.session_state.goal_question, messages)


def _answer(graph: FinnieGraph, query: str, profile: UserProfile, portfolio: Portfolio | None,
            messages: list[dict]) -> None:
    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    _stream_reply(graph.ask_stream(st.session_state.session_id, query, profile=profile, portfolio=portfolio),
                  messages)


def _stream_reply(events, messages: list[dict]) -> None:
    """Show a streamed reply, or rerun to show the form if the turn paused to ask a question."""
    with st.chat_message("assistant"):
        status = st.empty()
        body = st.empty()
        status.caption("Thinking...")
        text, reply = "", None
        for event in events:
            if event.type == "status":
                status.caption(event.text)
            elif event.type == "token":
                text += event.text
                body.markdown(text + " ▌")
            elif event.type == "interrupt":
                st.session_state.goal_question = event.data
                st.rerun()
            elif event.type == "final":
                reply = event.reply
        status.empty()
        body.markdown(reply.content)  # the final reply is authoritative (includes the disclaimer)
        _render_reply_extras(reply.agents, reply.sources)

    messages.append(
        {"role": "assistant", "content": reply.content, "agents": reply.agents, "sources": reply.sources}
    )


def _ask_goal_question(graph: FinnieGraph, question: dict, messages: list[dict]) -> None:
    """The Goal Planning Agent's paused question: how much of the portfolio counts toward the goal."""
    with st.chat_message("assistant"):
        st.markdown(question["question"])
        with st.form("goal_question_form", border=False):
            amount = st.number_input("Amount from your portfolio", min_value=0.0,
                                     max_value=float(question["portfolio_value"]), value=0.0, step=500.0,
                                     format="%.0f")
            remember = st.checkbox("Save this to my goal", help="Finnie won't ask again for this goal. "
                                   "You can change it on the My account page.")
            with st.container(horizontal=True):
                count = st.form_submit_button("Count it", type="primary", icon=":material/check:")
                skip = st.form_submit_button("Skip", icon=":material/skip_next:")
    if not (count or skip):
        return

    amount = amount if count else 0.0
    del st.session_state.goal_question
    if count and remember:
        _save_portfolio_amount(question["goal"], amount)
    messages.append({"role": "assistant", "content": question["question"]})
    choice = (f"Count ${amount:,.0f} of my portfolio toward {question['goal']}." if amount
              else f"Don't count my portfolio toward {question['goal']}.")
    messages.append({"role": "user", "content": choice})
    with st.chat_message("user"):
        st.markdown(choice)
    _stream_reply(graph.resume_stream(st.session_state.session_id, {"amount": amount}), messages)


def _save_portfolio_amount(goal_name: str, amount: float) -> None:
    """Store the amount on the signed-in user's goal so Finnie doesn't ask again."""
    profile = st.session_state.get("profile")
    if profile is None:
        return
    goals = [g.model_copy(update={"from_portfolio": amount}) if g.name == goal_name else g for g in profile.goals]
    updated = profile.model_copy(update={"goals": goals})
    try:
        get_user_store().save_profile(current_account().id, updated)
    except UserStoreUnavailable as exc:
        st.toast(str(exc), icon=":material/error:")
        return
    st.session_state.profile = updated
    st.toast(f"Saved to your {goal_name} goal", icon=":material/save:")
