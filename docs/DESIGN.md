# Finnie — Implementation & Design

How Finnie is built and why. Updated as components land. ✅ implemented · ⏳ planned

## 1. Architecture

A LangGraph router picks one or more specialist agents; they run in parallel, and their answers are merged, compliance-checked and shown in Streamlit.

```
user ─► router ─┬─► finance_qa ──┐
                ├─► portfolio ───┤
                ├─► market ──────┼─► synthesizer ─► compliance_guard ─► UI
                ├─► goal_planning┤
                ├─► news ────────┤
                └─► tax ─────────┘
```

| Component | Location | Status |
|---|---|---|
| Models, workflow state | `src/core/models.py`, `src/workflow/state.py` | ✅ |
| Router | `src/workflow/router.py` | ✅ |
| `BaseAgent`, prompts, Finance Q&A agent | `src/agents/` | ✅ |
| RAG: loader, chunker, FAISS index, retriever; 14 KB articles | `src/rag/`, `src/data/knowledge_base/` | ✅ |
| Market data provider, other 5 agents | `src/data/`, `src/agents/` | ⏳ |
| Graph (`FinnieGraph`), synthesizer, compliance guard, memory | `src/workflow/graph.py`, `nodes.py` | ✅ |
| Streamlit UI: Chat (live, streaming), Portfolio and Market tabs (static sample data) | `src/web_app/` | ✅ |
| Docker image + docker-compose (non-root, health check, secrets at run time) | `Dockerfile`, `docker-compose.yml` | ✅ |
| AWS deployment: ECR + ECS Fargate + ALB via CDK, CI/CD pipeline | — | ⏳ |
| MCP server (optional) | `mcp_server/` | ⏳ |

**Stack:** LangGraph (parallel nodes, reducers, checkpointing) · any LangChain chat model, Gemini Flash by default, with model/temperature configurable per agent (`AGENT_LLM`) · FAISS · Alpha Vantage behind a provider interface · Streamlit · Pydantic v2.

## 2. Key design decisions

**2.1 Hybrid tool use.** Agents pre-fetch the obvious data in code; Market and News also give the LLM tools for anything extra, in a loop capped at 3 rounds.
- *"How is AAPL doing?"* → AAPL quote pre-fetched → **1 LLM call**.
- *"Compare AAPL with its competitors"* → LLM calls `get_quote("MSFT")`, `get_quote("GOOGL")` → **2 calls**.

| Agent | Pre-fetched | LLM tools |
|---|---|---|
| Market | quotes/trends, or index snapshot | `get_quote`, `get_price_history` |
| News | headlines | `get_news` |
| Portfolio, Goal Planning | metrics / projection | — |
| Finance Q&A, Tax | — (RAG only) | — |

**2.2 Router.** Structured output (`RouterDecision`: agents, reason, entities), last 6 messages for context; falls back to keywords, then `finance_qa`.
- *"How is VTI doing and does it fit my retirement goal?"* → `["market", "goal_planning"]`, `tickers=["VTI"]`.

**2.3 `BaseAgent` template.** `run()` = `retrieve_context → prefetch_data → build_messages → generate → build_response`. Subclasses override only what they need; retries, error isolation, personalization and citations are shared.
- `FinanceQAAgent` overrides two methods. If its LLM fails, it returns a fallback message and other agents still answer.

**2.4 Numbers from code, explanations from the LLM.** Metrics, projections and statistics are computed in Python and passed to the LLM verbatim.
- "$500/month at 6% for 30 years ≈ $500k" is calculated; the LLM explains it.

**2.5 Scoped RAG behind an interface.** Agents use a `Retriever` protocol (testable before FAISS exists) and search only their categories.
- Market cites a "Moving averages" article for the concept but takes the price from market data.

**2.6 Layered compliance.** Shared `COMPLIANCE_RULES` in every prompt, plus a `compliance_guard` node that adds a caveat to advice-like phrasing and appends the disclaimer (kept out of conversation memory so the LLM doesn't repeat it).
- *"Should I buy TSLA?"* → how to evaluate a stock, no recommendation, disclaimer.

**2.7 Pydantic at boundaries, TypedDict for state.**
- A CSV row with `quantity = -5` is rejected by `Holding` before any calculation.

## 3. Agent communication

Agents never call each other; they read and write the shared `FinnieState`. Nodes return only the keys they change.

| Key | Written by | Merge |
|---|---|---|
| `user_query`, `user_profile`, `portfolio` | UI | overwrite |
| `messages` | memory | `add_messages` (append) |
| `route`, `route_reason`, `extracted_entities` | router | overwrite |
| `agent_outputs` | agents | `operator.add` (parallel-safe append) |
| `final_response`, `sources` | synthesizer / guard | overwrite |
| `errors` | any node | `operator.add` |

The UI calls `FinnieGraph.ask()` (complete reply) or `ask_stream()` (status → answer tokens → final reply). Only the step writing the user-facing answer streams: the single agent, or the synthesizer when several answered; the first words appear after ~2.2s.

Fan-out uses conditional edges: `select_agents` returns node names, LangGraph runs them in one superstep, and the synthesizer runs once after all finish.

**Example, end to end:** *"How is VTI doing, and does it fit my retirement goal?"*
1. Router → `market` + `goal_planning` (1 call).
2. In parallel: Market pre-fetches VTI and explains it (1 call); Goal Planning computes the projection and explains it (1 call).
3. Synthesizer merges both (1 call); guard adds the disclaimer.

Total **4 LLM calls**. *"VTI vs VOO?"* with only VTI extracted adds one tool round → 5.

## 4. Error handling

| Failure | Result |
|---|---|
| Router LLM fails | 3 attempts → keyword classifier → `finance_qa` |
| Agent LLM fails or returns empty | 3 attempts → fallback message for that agent only |
| Tool fails / unknown tool | Error sent to the LLM, which says the data was unavailable |
| LLM keeps requesting tools | Stopped after 3 rounds, final answer forced |
| No relevant KB match (score < 0.62) | Hedged answer, labelled as not from the knowledge base |
| Market API down (⏳) | Retry → cache → mock, with a freshness label |

## 5. RAG

- ✅ 14 Markdown articles (all 12 categories) with front-matter, verified against Investor.gov, IRS and other primary sources.
- ✅ Each article is a Markdown file with a YAML front-matter header (`title`, `category`, `source_name`, `source_url`, `level`, `last_reviewed`), parsed with `python-frontmatter` in `src/rag/loader.py`; the metadata is attached to every chunk for category filtering and citations.
- ✅ Split by `##` section, then by size (1,500 chars); each chunk is prefixed with "title > section" → 88 chunks.
- ✅ `gemini-embedding-001` vectors in FAISS with cosine similarity; a manifest records the embedding model, and the retriever refuses a mismatched index.
- ✅ Top-5 chunks, category filters, cutoff 0.62 (calibrated: on-topic 0.66–0.79, off-topic 0.50–0.55).
- ✅ Short follow-ups reuse the previous question: *"what about fees?"* → searches *"What is an ETF? what about fees?"*.
- ⏳ Grow to 50–100 articles plus a glossary.

## 6. Performance

- LLM calls = 1 router + 1 per agent + 1 synthesizer (if >1 agent); tool rounds add at most 3 per agent.
- Parallel agents: latency ≈ slowest agent.
- Bounded prompts: 6 history messages, 5 chunks.
- ⏳ 30-minute market cache, index loaded once, rate limiting, benchmarks in `docs/benchmarks.md`.

## 7. Open items

- Router test set: 30+ labelled queries, ≥ 90% accuracy.
- Confirm return (4/6/8%) and inflation (3%) assumptions; document portfolio score formulas.
