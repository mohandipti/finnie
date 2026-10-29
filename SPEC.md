# Finnie — AI Finance Assistant: Specification

> **Status:** Draft v1.0 · **Date:** 2026-10-26
> **Sources:** Capstone *Problem Statement* doc + *Project Milestones & FAQs* doc (Applied Agentic AI for SWEs).
> **Purpose:** Single source of truth for spec-driven development. Each requirement has an ID (`FR-*`, `NFR-*`) and acceptance criteria (AC) so it can be implemented, tested and traced to a grading criterion.

---

## 0. How to use this spec

- Implement in the order given in §14 (Milestones). Each milestone lists the requirement IDs it covers.
- **MUST** = needed for a passing submission. **SHOULD** = expected / graded. **MAY** = optional / bonus.
- Where the source docs allow a choice of technology, this spec picks a **default** (§3) and marks it as a decision. Changing a default is allowed, but record the reason in `docs/DESIGN.md` (the FAQ requires that technology choices be documented).
- Anything marked **[ASSUMPTION]** is not in the source docs. It is a reasonable default that you can revise.

---

## 1. Product overview

### 1.1 Problem
Financial literacy keeps many would-be investors out of the market. Complex terminology, too many choices and fear of mistakes stop beginners from building wealth through investing.

### 1.2 Solution
**Finnie** is a multi-agent conversational assistant that offers personalized, jargon-free financial **education** (not advice). It combines:
- Specialized agents per domain (Q&A, portfolio, market, goals, news, tax).
- RAG over a curated, cited financial knowledge base.
- Real-time market data so users can see concepts applied to real data.
- Conversation memory and a user profile (knowledge level, goals, risk tolerance), so guidance gets more sophisticated as the user learns.

### 1.3 Target users
Beginner and early-stage retail investors. They need plain-language explanations, help understanding their own holdings, and goal planning that takes their risk appetite into account.

### 1.4 Guiding principles
1. **Education, not advice.** Every response must avoid specific buy/sell recommendations and carry a disclaimer (regulatory compliance).
2. **Grounded and cited.** Factual claims come from the knowledge base or live data, with source attribution (user trust).
3. **Conservative by default.** When unsure, say so and recommend consulting a licensed professional.
4. **Graceful degradation.** API or LLM failures produce cached or mock data plus a clear notice, never a crash.
5. **Accessible.** Simple language, adjusted to the user's knowledge level.

### 1.5 Out of scope (v1)
Authentication and user accounts, brokerage or trade execution, voice and mobile clients, crypto, international exchanges, fine-tuned models, Monte Carlo simulation. See §16 for future directions.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| Agent | A LangGraph node with one domain responsibility, a defined input/output contract and its own prompt/tools |
| Router | The workflow node that classifies a query and selects one or more agents |
| KB | Knowledge base: the 50–100 curated articles plus a glossary, indexed in FAISS |
| Session | A Streamlit session. Holds conversation history and user profile in `st.session_state` |
| Freshness indicator | UI label showing when market data was fetched and whether it is live, cached or mock |

---

## 3. Technology decisions

| Concern | Options allowed by brief | **Default (decision)** | Notes |
|---|---|---|---|
| Orchestration | LangChain / LangGraph /
| LLM | Gemini  | **Provider-agnostic wrapper; default Google Gemini Flash** | FAQ recommends Gemini 2.0 Flash (free tier ≈60 req/min). Provider and model are set in `config.yaml` |
| Embeddings | — | **[ASSUMPTION]** Same provider's embedding model, or `sentence-transformers/all-MiniLM-L6-v2` for offline use | Configurable |
| Vector DB | FAISS / Chroma / Pinecone | **FAISS** (local, persisted to disk) | Milestone 6 names FAISS |
| Market data | Alpha Vantage | ** Alpha Vantage ** behind one `MarketDataProvider` interface | Milestone 8 names Alpha Vantage, so both are implemented |
| UI | Streamlit  | **Streamlit**, multi-tab |
| State / memory | LangGraph  | **LangGraph state + `MemorySaver` checkpointer keyed by session id**, mirrored in `st.session_state` | |
| Charts | — | **[ASSUMPTION]** Plotly | Pie, bar, line, projection charts |
| Config | YAML / env vars | `config.yaml` + `.env` (secrets) via `pydantic-settings` | |
| Testing | — | `pytest`, `pytest-cov`, `pytest-mock` | Target ≥ 80% coverage |
| Language | — | **[ASSUMPTION]** Python 3.11+ | |
| MCP (optional) | — | Official `mcp` Python SDK, stdio transport | 

---

## 4. System architecture

### 4.1 Data flow
```
User (Streamlit) ──► Workflow Router ──► Agent(s) ──► RAG retrieval / Tools (market data, calculators)
                                             │
                                             ▼
                                   LLM processing ──► Response Synthesizer ──► Compliance Guard ──► UI
```

### 4.2 LangGraph topology
```
START
  │
  ▼
[load_context]          # merge session profile + history into state
  │
  ▼
[router]                # classify intent(s) → state.route (list of agent names)
  │  conditional edges (fan-out; parallel when >1 agent)
  ├──► [finance_qa]
  ├──► [portfolio]
  ├──► [market]
  ├──► [goal_planning]
  ├──► [news]
  └──► [tax]
          │ (all converge)
          ▼
[synthesizer]           # merge agent outputs into one answer when >1 agent ran; pass-through otherwise
  │
  ▼
[compliance_guard]      # enforce disclaimer, strip advice-like phrasing, attach sources
  │
  ▼
[update_memory]         # persist turn, update profile signals
  │
  ▼
END
```
Any node error is caught and routed to `[fallback]`. The fallback node returns a safe, generic response, falls back to `finance_qa` when it can, and records the error in `state.errors`.

### 4.3 Directory structure (prescribed by brief)
```
ai_finance_assistant/
├── src/
│   ├── agents/          # base_agent.py + one module per agent
│   ├── core/            # llm.py (provider wrapper), config.py, models.py (pydantic), exceptions.py, disclaimers.py
│   ├── data/            # knowledge_base/ (articles, glossary), sample_portfolios/, market providers + cache
│   ├── rag/             # loader.py, chunker.py, embedder.py, index.py (FAISS), retriever.py
│   ├── web_app/         # app.py (Streamlit entry), tabs/ (chat, portfolio, market, goals), components/
│   ├── utils/           # logging.py, rate_limiter.py, retry.py, formatting.py
│   └── workflow/        # state.py, router.py, graph.py, nodes.py
├── tests/               # unit/, integration/, fixtures/
├── mcp_server/          # [optional] server.py
├── docs/                # DESIGN.md, architecture diagrams
├── config.yaml
├── .env.example
├── requirements.txt
├── Dockerfile           # [optional]
└── README.md
```

---

## 5. Core data models

Use Pydantic models in `src/core/models.py`. All agent I/O must use these types.

```python
class UserProfile(BaseModel):
    session_id: str
    knowledge_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = None
    goals: list[FinancialGoal] = []
    interests: list[str] = []            # e.g. ["ETFs", "retirement"], inferred from conversation

class Holding(BaseModel):
    ticker: str
    quantity: float                      # > 0
    cost_basis: float | None = None      # per unit
    asset_type: Literal["stock","etf","mutual_fund","bond","cash","other"] | None = None  # auto-detected if None

class Portfolio(BaseModel):
    holdings: list[Holding]

class FinancialGoal(BaseModel):
    name: str                            # e.g. "Retirement", "House down payment"
    target_amount: float
    current_savings: float = 0
    monthly_contribution: float = 0
    horizon_years: float
    risk_tolerance: Literal["conservative","moderate","aggressive"]

class Source(BaseModel):
    title: str
    url: str | None
    category: str | None
    snippet: str | None
    kind: Literal["kb","market_api","news_api"]

class AgentResponse(BaseModel):
    agent: str
    content: str                         # markdown
    sources: list[Source] = []
    data: dict | None = None             # structured payload for UI charts (metrics, projections, quotes)
    confidence: float | None = None      # 0–1
    error: str | None = None

class Quote(BaseModel):
    ticker: str; price: float; change: float; change_pct: float
    currency: str = "USD"; as_of: datetime
    source: Literal["alpha_vantage","cache","mock"]
```

### 5.1 Workflow state (`src/workflow/state.py`)
```python
class FinnieState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]   # conversation history with reducer function
    user_query: str
    user_profile: UserProfile
    portfolio: Portfolio | None
    route: list[str]                                        # agents selected by router
    route_reason: str
    agent_outputs: Annotated[list[AgentResponse], operator.add]   # reducer to support parallel fan-out
    retrieved_docs: list[Source]
    final_response: str
    sources: list[Source]
    errors: Annotated[list[str], operator.add]
```

---

## 6. Functional requirements: agents

### FR-AG-0 Base agent (MUST)
An abstract `BaseAgent` in `src/agents/base_agent.py` with:
- `name: str`, `description: str` (the router uses this), `system_prompt: str`.
- `run(state: FinnieState) -> AgentResponse` (template method): build context → optionally retrieve (RAG) → pre-fetch data in code → call LLM (single call, or a bounded tool-calling loop when the agent exposes tools) → post-process. See `docs/DESIGN.md` §2.1 for the hybrid tool-use design.
- Shared helpers: `retrieve(query, categories)`, `adapt_to_level(profile)`, `format_sources()`, error wrapping into `AgentResponse(error=...)`.
- **AC:** Every concrete agent subclasses `BaseAgent`. An exception inside an agent never propagates out of `run()`.

### FR-AG-1 Finance Q&A Agent (MUST, core)
- **Scope:** General financial education (concepts, terminology, investment types, compound interest, diversification and similar topics).
- **Uses:** RAG (all KB categories plus glossary).
- **Behavior:** Answers in plain language at the user's `knowledge_level`, defines jargon inline, cites KB sources. If retrieval relevance is below the threshold (§8), it says the KB doesn't cover the topic and gives a hedged general answer.
- **AC:**
  - "What is an ETF?" → answer cites ≥ 1 KB source from the `etfs` category.
  - A beginner profile gets no unexplained jargon. An advanced profile gets more technical depth.
  - "Should I buy TSLA?" → educational framing with no buy/sell recommendation, plus a disclaimer.

### FR-AG-2 Portfolio Analysis Agent (MUST, core)
- **Input:** `Portfolio` from the UI form, CSV upload or chat (e.g. "I have 10 AAPL and 5 VTI").
- **Tools:** Market data (current prices), asset-type and expense-ratio lookup, metrics calculator (pure Python, deterministic).
- **Metrics** (from FAQ; computed in code, not by the LLM):

| Metric | Definition |
|---|---|
| Total value | Σ quantity × current price |
| Allocation % | Per holding and per asset type |
| Gain/loss | Per holding and total, when cost basis is given |
| Weighted expense ratio | Σ(weight × expense ratio) for funds; 0 for individual stocks |
| Diversification score (0–100) | **[ASSUMPTION]** `100 × (1 − HHI)` normalized, where HHI = Σ weight². Bonus/penalty for number of asset types and sector concentration when available |
| Risk level | **[ASSUMPTION]** Weighted score from asset-type risk weights (cash 1, bond 2, ETF/mutual fund 3, stock 4, other 5) mapped to Low / Medium / High |

- **Output:** `AgentResponse.data` = metrics dict for dashboard charts. `content` = LLM explanation of the metrics in plain language, with *educational* observations (e.g. "70% in one stock means higher concentration risk") and no prescriptive trades.
- **AC:**
  - Metric functions are unit-tested against fixture portfolios with known expected values.
  - Unknown ticker → that holding is flagged and the remaining holdings are still analyzed.
  - Empty portfolio → friendly validation message.

### FR-AG-3 Market Analysis Agent (MUST, core)
- **Scope:** Real-time quotes, price history, simple trends (e.g. 1M/6M/1Y change, 50/200-day moving averages, index snapshot for S&P 500 / NASDAQ / Dow).
- **Tools:** `MarketDataProvider` (§9).
- **Uses:** RAG, as a supplement to the live data. Retrieves short concept explanations from the `market_concepts`, `risk` and `glossary` categories for the concepts the response mentions (e.g. moving average, volatility, index). This follows the brief's data flow (Agent → RAG Retrieval → LLM) while live data remains the source of facts.
- **Behavior:** Explains *what the data shows* and links it to concepts ("this is what volatility looks like"), citing KB sources for concept explanations. Every response includes the data timestamp and source (freshness). Prices and statistics always come from the provider, never from the KB.
- **AC:**
  - "How is AAPL doing?" → current price, % change, trend summary, and an `as_of` timestamp.
  - A response that explains a concept (e.g. moving average) cites ≥ 1 KB source; retrieval is restricted to the categories above.
  - With no relevant KB match (or no retriever), the response still returns the market data, without concept citations.
  - With the provider down, the response uses cached or mock data and says so explicitly.

### FR-AG-4 Goal Planning Agent (MUST, core)
- **Input:** `FinancialGoal` (from the Goals tab or extracted from chat). Defaults to the profile's risk tolerance.
- **Projection algorithm** (deterministic code):
  - Future value with monthly compounding: `FV = PV(1+r)^n + PMT × [((1+r)^n − 1)/r]`, where `r` = annual return / 12 and `n` = months.
  - **[ASSUMPTION]** Expected annual returns by risk profile, configurable in `config.yaml`: conservative 4%, moderate 6%, aggressive 8%. Also show a low/high band (±2%).
  - Solve for the monthly contribution needed to reach the target, and for the years needed at the current contribution.
  - Optional inflation adjustment (default 3%, configurable).
- **Behavior:** Explains the projection, the gap to target and trade-offs (time vs. contribution vs. risk). Suggests *educational* example allocation ranges per risk profile, clearly labelled as illustrative.
- **AC:**
  - Projection math is unit-tested (including the `r = 0` edge case).
  - Changing the risk tolerance changes the projection.
  - Output includes `data.projection` (year-by-year series) for the chart.

### FR-AG-5 News Synthesizer Agent (SHOULD, "if time permits")
- **Scope:** Summarizes recent financial news for a ticker or topic and explains why it matters to a beginner.
- **Tools:** **[ASSUMPTION]** `yfinance` `Ticker.news` and/or Alpha Vantage `NEWS_SENTIMENT`.
- **AC:** Returns ≤ 5 summarized headlines with source links and publish dates. Gives a neutral contextualization with no price predictions.

### FR-AG-6 Tax Education Agent (SHOULD, "stretch")
- **Scope:** Explains tax concepts and account types (401(k), IRA vs. Roth IRA, HSA, capital gains short vs. long term, tax-loss harvesting concept, dividends).
- **Uses:** RAG (`taxes` category).
- **Behavior:** General education only. Must recommend a tax professional for personal situations. **[ASSUMPTION]** US-focused; states the jurisdiction in responses.
- **AC:** "Roth vs traditional IRA?" → comparison with KB citation and a tax disclaimer.

> **Priority (FAQ):** Quality over quantity. Q&A, Portfolio, Market and Goal agents are core. Implement at least 4–5 agents well. The target is all 6.

---

## 7. Functional requirements: workflow and orchestration

### FR-WF-1 Router (MUST)
- An LLM classifier with **structured output** (`{"agents": [...], "reason": str, "extracted": {...}}`) built from each agent's `name` and `description`.
- Can select **multiple agents** for compound queries (e.g. "How is VTI doing and is it a good fit for my retirement goal?" → `market` + `goal_planning`).
- Extracts entities into state: tickers, amounts, horizons, holdings.
- **Fallback:** If the LLM fails or returns an invalid label, a keyword/regex classifier runs. If that also gives nothing, route to `finance_qa`.
- **AC:** Routing accuracy ≥ 90% on a labelled test set of ≥ 30 queries (`tests/fixtures/routing_cases.yaml`), including malformed or empty queries.

### FR-WF-2 Parallel execution and synthesis (MUST)
- Multiple selected agents run in parallel (LangGraph fan-out using `Send` or conditional edges).
- The synthesizer merges outputs into one coherent answer, deduplicates sources and keeps each agent's structured `data` for the UI.

### FR-WF-3 Conversation memory and context (MUST)
- Full message history per session, using the LangGraph checkpointer keyed by `session_id` and mirrored in `st.session_state`.
- Follow-ups resolve references ("what about its expense ratio?" → the previously discussed ticker).
- **[ASSUMPTION]** Keep the last N=20 turns verbatim and summarize older turns to bound the token count.
- Profile signals (risk tolerance, goals, knowledge level) persist across turns, and later turns use them.
- **AC:** In a multi-turn integration test, turn 3 correctly uses an entity introduced in turn 1.

### FR-WF-4 Compliance guard (MUST)
- Appends the standard disclaimer (§11) to every response.
- Detects advice-like phrasing ("you should buy/sell", "guaranteed returns") and rewrites it or adds a caveat.
- Ensures `sources` are rendered.
- **AC:** Unit tests: an advice-like draft gets rewritten or caveated, and the disclaimer is always present.

### FR-WF-5 Error handling and fallback (MUST)
- Every node is wrapped. Errors are logged with the session id and node name and appended to `state.errors`.
- LLM call failures: retry with exponential backoff (3 attempts), then return a fallback message.
- The user never sees a stack trace.

---

## 8. Functional requirements: RAG and knowledge base

### FR-KB-1 Content curation (MUST)
- **50–100** financial education articles in `src/data/knowledge_base/articles/` as Markdown with YAML front-matter:
  ```yaml
  title: "What is an ETF?"
  category: etfs
  source_url: https://www.investopedia.com/...
  source_name: Investopedia
  level: beginner
  last_reviewed: 2026-10-26
  ```
- **Categories** (minimum): `basics`, `stocks`, `bonds`, `etfs`, `mutual_funds`, `diversification`, `risk`, `retirement`, `taxes`, `goal_planning`, `market_concepts`, `behavioral_finance`.
- **Glossary** of financial terms: `src/data/knowledge_base/glossary.yaml` (term, definition, related terms). **[ASSUMPTION]** ≥ 100 terms.
- **Reference sources** (from brief): Investopedia, Bogleheads wiki, SEC investor.gov, Modern Portfolio Theory basics. Write content in your own words or summarize it, and always record the source URL.
- **Sample data:** ≥ 3 sample portfolios (conservative, balanced, concentrated/aggressive) in `src/data/sample_portfolios/` as CSV/JSON.

### FR-RAG-1 Ingestion and indexing (MUST)
- Loader parses front-matter into metadata.
- Chunker: **[ASSUMPTION]** recursive character splitter, ~800 tokens with 100 overlap, splitting on headings first.
- Embed and store in FAISS, persisted to `data/index/`. A CLI command `python -m src.rag.index --rebuild` rebuilds the index.
- Glossary entries are indexed as their own small documents (`category: glossary`).

### FR-RAG-2 Retrieval (MUST)
- `retrieve(query, k=5, categories: list[str] | None)` supports **category-based metadata filtering**.
- **Relevance scoring:** Similarity score normalized to 0–1. Drop results below a configurable `min_score` (default **[ASSUMPTION]** 0.35).
- **[MAY]** MMR or a re-ranker for diversity.
- **AC:**
  - Filtering to `taxes` returns only tax chunks.
  - An off-topic query ("best pizza recipe") returns no results above the threshold.

### FR-RAG-3 Source attribution (MUST)
- Every RAG-based answer lists its sources (title, source name, URL) in a "Sources" section in the UI.

---

## 9. Functional requirements: real-time market data

### FR-MD-1 Provider abstraction (MUST)
```python
class MarketDataProvider(Protocol):
    def get_quote(self, ticker: str) -> Quote: ...
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame: ...
    def get_info(self, ticker: str) -> dict: ...   # name, asset type, sector, expense ratio if fund
    def get_news(self, ticker: str, limit: int = 5) -> list[dict]: ...
```
Implementations: `YFinanceProvider`, `AlphaVantageProvider`, `MockProvider` (fixtures). A `ResilientProvider` composite tries primary → secondary → cache → mock.

### FR-MD-2 Caching (MUST)
- TTL cache for quotes, history and info. **Default TTL 30 minutes** (FAQ), configurable.
- **[ASSUMPTION]** In-memory TTL cache with an optional disk cache (`diskcache`), so a stale value can still be served when every provider fails.

### FR-MD-3 Rate limits and failures (MUST)
- Client-side rate limiter per provider (token bucket). Limits are configurable, e.g. Alpha Vantage free tier.
- Exponential backoff with jitter on transient errors (max 3 retries).
- Invalid ticker → a typed `TickerNotFound` error that becomes a friendly message.

### FR-MD-4 Freshness indicators (MUST)
- Every `Quote` and response shows `as_of` and `source` (live / cached / mock). The UI shows a badge such as "Live · 2 min ago" or "Cached · 25 min ago" or "Sample data".

### FR-MD-5 Trend analysis (SHOULD)
- Compute % change over 1D/1M/6M/1Y, 50/200-day SMA, and **[ASSUMPTION]** annualized volatility. The Market agent explains these in plain language.

---

## 10. Functional requirements: user interface (Streamlit)

Clean and beginner-friendly, with **4 main tabs** (FAQ). Grading focuses more on the agent architecture than on UI polish.

### FR-UI-1 Chat tab (MUST)
- Conversational interface (`st.chat_message`, `st.chat_input`) with streaming responses where supported.
- Each response shows which agent(s) answered (badge), a collapsible "Sources" section and the disclaimer.
- Inline charts when an agent returns `data` (e.g. a mini price chart).
- Sidebar: profile settings (knowledge level, risk tolerance), "New conversation", example prompts.

### FR-UI-2 Portfolio tab (MUST)
- Input via an editable table (`st.data_editor`), CSV upload or "Load sample portfolio".
- Dashboard: KPI cards (total value, gain/loss, diversification score, risk level, weighted expense ratio), **allocation pie chart**, **allocation bar chart** by asset type, per-holding table.
- An "Explain my portfolio" button invokes the Portfolio agent and shows its narrative.

### FR-UI-3 Market tab (MUST)
- Index snapshot (S&P 500, NASDAQ, Dow), ticker search, **price trend line chart** with SMA overlays, period selector, freshness badge.
- **[SHOULD]** News panel (News agent).

### FR-UI-4 Goals tab (MUST)
- Goal form (name, target, current savings, monthly contribution, horizon, risk tolerance).
- **Goal projection chart** (expected path plus low/high band, target line), required-contribution calculation, and the agent's explanation.
- Supports multiple goals per session.

### FR-UI-5 Session handling (MUST)
- `st.session_state` holds session id, messages, profile, portfolio and goals. No authentication required.
- The profile and portfolio entered in the tabs are visible to the chat agents (shared state).

### FR-UI-6 Responsiveness and UX (SHOULD)
- Spinners and status during agent runs. Friendly error toasts. Charts are clearly labelled (titles, axis labels, currency formatting).

---

## 11. Compliance and trust requirements

| ID | Requirement |
|---|---|
| NFR-CMP-1 | Standard disclaimer on every response: *"Finnie provides educational information only and is not financial, investment, tax or legal advice. Consult a licensed professional before making financial decisions."* |
| NFR-CMP-2 | No specific buy/sell/hold recommendations for individual securities. No return guarantees. No price predictions. |
| NFR-CMP-3 | Cite sources for factual content (KB articles, data provider plus timestamp). |
| NFR-CMP-4 | Stick to widely accepted principles. If unsure or out of scope, say so. |
| NFR-CMP-5 | Tax content states the jurisdiction (US) and recommends a tax professional. |
| NFR-CMP-6 | A visible disclaimer banner in the app footer/sidebar. |

---

## 12. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-PERF-1 | Latency | **[ASSUMPTION]** p50 single-agent response < 5 s; multi-agent < 10 s (excluding cold start). Measure and record in benchmarks |
| NFR-PERF-2 | Efficiency | Market data served from cache within TTL. FAISS index loaded once per process (`st.cache_resource`) |
| NFR-REL-1 | Resilience | No unhandled exceptions reach the UI. API outages degrade to cached or mock data |
| NFR-REL-2 | LLM limits | Client-side rate limiting for the LLM (Gemini free tier ≈60 RPM) with backoff |
| NFR-OBS-1 | Logging | Structured logging (`logging` + JSON formatter) with session id, node and latency. Log level set in config |
| NFR-SEC-1 | Secrets | API keys only in `.env` (never committed). Ship `.env.example` |
| NFR-CFG-1 | Config | All tunables (models, TTLs, thresholds, return assumptions, rate limits, k, chunk sizes) live in `config.yaml`; env vars override them |
| NFR-QA-1 | Testing | ≥ 80% line coverage. Unit and integration tests. Edge cases (malformed queries, empty input, invalid tickers, API failures). LLM and market calls mocked in unit tests |
| NFR-MAINT-1 | Code quality | Modular, typed (type hints), docstrings on public functions. **[ASSUMPTION]** `ruff` + `black` |
| NFR-DEP-1 | Runnability | One-command local run (`streamlit run src/web_app/app.py`). **[MAY]** Docker + docker-compose for bonus |

---

## 13. Optional: MCP server (bonus, 5%)

### FR-MCP-1
- `mcp_server/server.py` exposes tools over the Model Context Protocol (stdio):
  - `get_stock_quote(ticker)`, `get_price_history(ticker, period)`
  - `analyze_portfolio(holdings: list[{ticker, quantity, cost_basis?}])`
  - `project_goal(target_amount, current_savings, monthly_contribution, horizon_years, risk_tolerance)`
  - `search_financial_knowledge(query, category?)`
- Reuses `src/` services. No duplicated logic.
- Documented with a Claude Desktop config snippet (`claude_desktop_config.json`) in the README.
- **AC:** Tools are listed and callable from Claude Desktop (or MCP Inspector).

---

## 14. Milestones and implementation plan

| # | Milestone | Key tasks | Requirements |
|---|---|---|---|
| 1 | Research and architecture | Study robo-advisors and edu platforms. Finalize §3 decisions. Scaffold repo structure, config, logging, LLM wrapper | §3, §4.3, NFR-CFG-1, NFR-OBS-1, NFR-SEC-1 |
| 2 | Knowledge base | 50–100 articles by category, glossary, sample portfolios | FR-KB-1 |
| 3 | Core agents | BaseAgent, Q&A (+RAG), Portfolio (+calculators), Market (+API) | FR-AG-0..3 |
| 4 | Advanced agents | Goal Planning (projection), News (if time), Tax (stretch), inter-agent tests | FR-AG-4..6 |
| 5 | Workflow orchestration | StateGraph, router, parallel fan-out, synthesizer, memory, fallback, compliance guard | FR-WF-1..5 |
| 6 | RAG system | FAISS, chunking/embedding, retrieval with scoring and filters, source attribution | FR-RAG-1..3 |
| 7 | UI | Chat, Portfolio dashboard, Market overview, Goals | FR-UI-1..6 |
| 8 | Real-time data | Alpha Vantage + yFinance, caching, rate limits, freshness indicators | FR-MD-1..5 |
| 9 | Testing and QA *(optional but graded 5%)* | Unit, integration, UAT, perf optimization | NFR-QA-1, NFR-PERF-* |
| 10 | Docs and deployment | README, DESIGN.md, demo video, artifacts, polish | §15 |
| 11 | MCP server *(stretch)* | MCP server + Claude Desktop integration + docs | FR-MCP-1 |

> **Recommended build order for spec coding:** 1 → 2 → 6 (RAG) → 3 → 8 (market data, needed by Portfolio/Market agents) → 5 → 4 → 7 → 9 → 10 → 11. Doc milestones 6 and 8 are dependencies of milestone 3, so pull them earlier.

---

## 15. Deliverables and submission checklist

### 15.1 Working prototype (MUST)
- [ ] ≥ 4 core agents working (target: all 6) with distinct capabilities
- [ ] Web interface for conversation
- [ ] Portfolio analysis of user input
- [ ] Real-time market data lookup
- [ ] Goal planning that takes risk appetite into account
- [ ] Error handling and logging throughout

### 15.2 Demo video (5–10 min)
- [ ] Multi-turn conversation that hits different agents (show agent badges and routing)
- [ ] Portfolio analysis demo
- [ ] Market data integration (with freshness indicator)
- [ ] Goal planning example
- [ ] A compound query showing agents working together

### 15.3 Documentation
- [ ] **README:** architecture overview (diagram), setup, API documentation (agent/tool interfaces), usage examples, troubleshooting
- [ ] **docs/DESIGN.md (technical design):** architecture decisions and rationale (including deviations from §3), agent communication protocol (state schema, router contract), RAG implementation details, performance considerations
- [ ] Inline docstrings

### 15.4 Deployment artifacts
- [ ] `requirements.txt`, `config.yaml`, `.env.example`
- [ ] Sample data for testing (portfolios, KB, mock market fixtures)
- [ ] Performance benchmarks (`docs/benchmarks.md`: latency per agent, cache hit rate, retrieval time)
- [ ] *(optional)* `Dockerfile` / `docker-compose.yml`

### 15.5 Tests *(optional per brief, but 5% of grade and 80% target)*
- [ ] Unit: each agent, calculators, router, retriever, providers, cache, compliance guard
- [ ] Integration: end-to-end graph runs with mocked LLM and market data, multi-turn memory
- [ ] Edge cases: malformed/empty queries, invalid tickers, API outage, rate limit hit

---

## 16. Evaluation criteria → spec traceability

| Criterion | Weight | Covered by |
|---|---|---|
| Multi-Agent Architecture | 10% | FR-AG-0..6 |
| LangGraph Workflow | 10% | FR-WF-1..5, §4.2, §5.1 |
| RAG Implementation | 8% | FR-KB-1, FR-RAG-1..3 |
| Real-time Data Integration | 7% | FR-MD-1..5 |
| MCP Server (bonus) | 5% | FR-MCP-1 |
| Streamlit Application | 10% | FR-UI-1..6 |
| Conversational Flow | 8% | FR-WF-3, FR-UI-1 |
| Data Visualization | 7% | FR-UI-2..4 |
| Educational Content | 8% | FR-KB-1, FR-AG-1, FR-AG-6 |
| Portfolio Analysis | 7% | FR-AG-2, FR-UI-2 |
| Market Intelligence | 5% | FR-AG-3, FR-AG-5, FR-MD-5 |
| Code Organization | 5% | §4.3, NFR-MAINT-1 |
| Documentation | 5% | §15.3 |
| Testing | 5% | NFR-QA-1, §15.5 |

> FAQ guidance: the multi-agent architecture and workflow orchestration matter most. Show sophisticated system design rather than UI polish. Docker or cloud-readiness earns bonus points.

---

## 17. Future directions (not in v1)
- **Features:** voice interface, mobile apps, Monte Carlo and risk modeling, community/social learning, brokerage integration.
- **Technical:** multi-modal input (chart and document parsing), ML personalization engine, microservices, real-time multi-user collaboration.
- **Market:** international exchanges, crypto education, small-business finance, partnerships with educators.
- **AI:** fine-tuned domain models, behavioral-bias detection, explainable AI, continuous learning from interactions.

---

## 18. Open questions / decisions to confirm
1. LLM provider: Gemini Flash (FAQ recommendation) vs. Claude Sonnet vs. OpenAI. Default is a configurable wrapper with Gemini.
2. Embedding model: provider-hosted or local `sentence-transformers` (offline, free).
3. Return assumptions for goal projections (4/6/8%) and inflation (3%): confirm or replace with sourced figures.
4. Diversification and risk score formulas (§FR-AG-2) are this spec's own definitions. Document them in DESIGN.md.
5. News source for the News agent (yfinance news vs. Alpha Vantage NEWS_SENTIMENT).
6. Jurisdiction for tax content (US assumed).
