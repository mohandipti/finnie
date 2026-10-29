"""Chat tab: a conversation with Finnie, streamed from the LangGraph workflow."""

import streamlit as st

from src.core.models import UserProfile
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


def render(graph: FinnieGraph | None, profile: UserProfile, startup_error: str | None) -> None:
    if startup_error:
        st.error(startup_error)
        return

    messages: list[dict] = st.session_state.messages
    if not messages:
        st.markdown("Ask me anything about personal finance and investing basics. Try one of these:")
        columns = st.columns(2)
        for i, prompt in enumerate(EXAMPLE_PROMPTS):
            if columns[i % 2].button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_reply_extras(message.get("agents", []), message.get("sources", []))

    query = st.chat_input("Ask Finnie about money and investing") or st.session_state.pop("pending_prompt", None)
    if not query:
        return

    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status = st.empty()
        body = st.empty()
        status.caption("Thinking...")
        text, reply = "", None
        for event in graph.ask_stream(st.session_state.session_id, query, profile=profile):
            if event.type == "status":
                status.caption(event.text)
            elif event.type == "token":
                text += event.text
                body.markdown(text + " ▌")
            elif event.type == "final":
                reply = event.reply
        status.empty()
        body.markdown(reply.content)  # the final reply is authoritative (includes the disclaimer)
        _render_reply_extras(reply.agents, reply.sources)

    messages.append(
        {"role": "assistant", "content": reply.content, "agents": reply.agents, "sources": reply.sources}
    )
