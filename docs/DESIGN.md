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
| `BaseAgent`, prompts and all six agents: Finance Q&A, Market, Portfolio, News, Tax and Goal Planning | `src/agents/` | ✅ |
| RAG: loader, chunker, FAISS index, retriever; 80 KB articles | `src/rag/`, `src/data/knowledge_base/` | ✅ |
| Market data provider (Alpha Vantage, cached) | `src/data/market_data.py` | ✅ |
| News provider (Google News via SerpAPI, cached) | `src/data/news_data.py` | ✅ |
| Goal projections (assumed returns, low/high range, monthly amount needed, time to target) | `src/core/goal_projection.py` | ✅ |
| Graph (`FinnieGraph`), synthesizer, compliance guard, memory (conversations in MongoDB, 30-day expiry) | `src/workflow/graph.py`, `nodes.py` | ✅ |
| Streamlit UI: Chat (streaming), Portfolio and Market tabs (live prices, fetched only while the tab is open) | `src/web_app/` | ✅ |
| Docker image + docker-compose (non-root, health check, secrets at run time) | `Dockerfile`, `docker-compose.yml` | ✅ |
| AWS infrastructure (CDK), deployed: Docker on EC2 t2.micro (us-east-1), ECR, Secrets Manager, CloudWatch, IP allow-list | `infra/` | ✅ |
| CI/CD pipeline (GitHub Actions + OIDC) | — | ⏳ |
| MCP server (optional) | `mcp_server/` | ⏳ |

**Stack:** LangGraph (parallel nodes, reducers, checkpointing) · any LangChain chat model, Gemini Flash by default, with model/temperature configurable per agent (`AGENT_LLM`) · FAISS · Alpha Vantage behind a provider interface · Streamlit · Pydantic v2.

## 2. Key design decisions

**2.1 Hybrid tool use.** Agents pre-fetch the obvious data in code; Market and News also give the LLM tools for anything extra, in a loop capped at 3 rounds.
- *"How is AAPL doing?"* → AAPL quote pre-fetched → **1 LLM call**.
- *"Compare AAPL with its competitors"* → LLM calls `get_quote("MSFT")`, `get_quote("GOOGL")` → **2 calls**.

| Agent | Pre-fetched | LLM tools |
|---|---|---|
| Market | quote + trend for up to 3 extracted tickers | `get_quote`, `get_price_trend` |
| News | Google News headlines for up to 2 extracted tickers (`"<TICKER> stock"`) | `get_news` (any company or topic) |
| Portfolio | holdings priced at latest quotes + `compute_metrics` (`src/core/portfolio_metrics.py`) | — |
| Goal Planning | projection for the saved goal(s) the question is about; asks first how much of the portfolio counts (2.8) | `project_goal` (numbers in the message, or what-ifs) |
| Finance Q&A | — (RAG only, whole knowledge base) | — |
| Tax | — (RAG only: tax and retirement articles first, then the whole knowledge base) | — |

**2.2 Router.** Structured output (`RouterDecision`: agents, reason, entities), last 6 messages for context; falls back to keywords, then `finance_qa`.
- *"How is VTI doing and does it fit my retirement goal?"* → `["market", "goal_planning"]`, `tickers=["VTI"]`.

**2.3 `BaseAgent` template.** `run()` = `retrieve_context → prefetch_data → build_messages → generate → build_response`. Subclasses override only what they need; retries, error isolation, personalization and citations are shared.
- `FinanceQAAgent` overrides two methods; `MarketAgent` adds `prefetch_data` and `get_tools`; `PortfolioAgent` analyzes holdings named in the message, else the saved portfolio; with neither, it explains how to add holdings; `NewsAgent` also overrides `build_response` so each article is a linked source. If an agent's LLM fails, it returns a fallback message and other agents still answer.

**2.4 Numbers from code, explanations from the LLM.** Metrics, projections and statistics are computed in Python and passed to the LLM verbatim.
- "$500/month at 6% for 30 years ≈ $500k" is calculated; the LLM explains it.

**2.5 Scoped RAG behind an interface.** Agents use a `Retriever` protocol (testable before FAISS exists) and search only their categories.
- Market cites a "Moving averages" article for the concept but takes the price from market data.

**2.6 Layered compliance.** Shared `COMPLIANCE_RULES` in every prompt, plus a `compliance_guard` node that adds a caveat to advice-like phrasing and appends the disclaimer (kept out of conversation memory so the LLM doesn't repeat it).
- *"Should I buy TSLA?"* → how to evaluate a stock, no recommendation, disclaimer.

**2.7 Pydantic at boundaries, TypedDict for state.**
- A CSV row with `quantity = -5` is rejected by `Holding` before any calculation.

**2.8 Human in the loop for goal projections.** A portfolio can't simply be added to a goal: it may fund several goals, or sit in retirement accounts. So when a question is about one saved goal, the user has a portfolio, and the goal's `from_portfolio` is blank, the Goal Planning node calls LangGraph `interrupt()` before any retrieval or LLM call.
- The chat shows the portfolio value and asks how much counts toward the goal (number box, *Count it* / *Skip*, optional *Save this to my goal*).
- `resume_stream(session_id, {"amount": x})` finishes the turn; the amount is capped at the portfolio value and added to the goal's current savings.
- The answer is kept in `goal_portfolio_answers` for the session; *Save* stores it as the goal's `from_portfolio`, editable on My account, so Finnie stops asking.
- The pause is at the start of the agent's own node, not a separate node before it: an extra step on one branch would make the synthesizer run twice. On resume LangGraph re-runs the node from the start; only the (cached) portfolio pricing runs again, and agents that ran alongside it keep their results.
- A new chat message while a question is open counts as *Skip*.
- The paused turn lives in the checkpointer (MongoDB, section 7), so it survives a restart and can be resumed by another app instance.

## 3. Agent communication

Agents never call each other; they read and write the shared `FinnieState`. Nodes return only the keys they change.

| Key | Written by | Merge |
|---|---|---|
| `user_query`, `user_profile`, `portfolio` | UI | overwrite (`portfolio` every turn, so a deleted portfolio is cleared) |
| `messages` | memory | `add_messages` (append) |
| `route`, `route_reason`, `extracted_entities` | router | overwrite |
| `agent_outputs` | agents | `operator.add` (parallel-safe append) |
| `final_response`, `sources` | synthesizer / guard | overwrite |
| `errors` | any node | `operator.add` |
| `goal_portfolio_answers` | Goal Planning (after the user answers) | overwrite; kept for the session |

The UI calls `FinnieGraph.ask()` (complete reply) or `ask_stream()` (status → answer tokens → final reply, or an `interrupt` event with a question, finished by `resume_stream()`). Only the step writing the user-facing answer streams: the single agent, or the synthesizer when several answered; the first words appear after ~2.2s.

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
| Market API down or daily limit reached | Last good cached copy, labelled "cached"; with none, the agent says the data is unavailable |
| News search fails or monthly quota used up | Last good result for that search, labelled "cached"; with none, the agent says the news is unavailable |

## 5. RAG

- ✅ 80 Markdown articles (all 12 categories; the largest are taxes 17, basics 12, stocks 11 and retirement 11) with front-matter, written from and verified against primary sources: the SEC (Investor.gov, sec.gov), FINRA, the IRS, the Federal Reserve, the CFPB, the FDIC, SIPC, TreasuryDirect, the Department of Labor, the Social Security Administration, S&P Dow Jones Indices and Fidelity.
- ✅ Each article is a Markdown file with a YAML front-matter header (`title`, `category`, `source_name`, `source_url`, `level`, `last_reviewed`), parsed with `python-frontmatter` in `src/rag/loader.py`; the metadata is attached to every chunk for category filtering and citations.
- ✅ Split by `##` section, then by size (1,500 chars); each chunk is prefixed with "title > section" → 475 chunks.
- ✅ `gemini-embedding-001` vectors in FAISS with cosine similarity; a manifest records the embedding model, and the retriever refuses a mismatched index.
- ✅ Top-5 chunks, category filters, cutoff 0.62 (calibrated: on-topic 0.66–0.79, off-topic 0.50–0.55).
- ✅ Short follow-ups reuse the previous question: *"what about fees?"* → searches *"What is an ETF? what about fees?"*.
- ⏳ Grow to about 100 articles plus a glossary.

## 6. Performance

- LLM calls = 1 router + 1 per agent + 1 synthesizer (if >1 agent); tool rounds add at most 3 per agent.
- Parallel agents: latency ≈ slowest agent.
- Bounded prompts: 6 history messages, 5 chunks.
- ✅ Market data (`src/data/market_data.py`): Alpha Vantage quotes (`GLOBAL_QUOTE`, cached 30 minutes) and weekly adjusted price history (`TIME_SERIES_WEEKLY_ADJUSTED`, 5 years, cached 12 hours; full daily history is a premium feature). `summarize_trend` gives 1M/6M/1Y change, 10- and 40-week averages (≈ 50/200-day) and annualized volatility. Requests are spaced 1.5 s apart (one service is shared by the agents, behind a lock), and the last good copy is kept as a fallback when the free tier's ~25 requests/day run out. The Portfolio and Market tabs only fetch prices while open. The Market tab shows the S&P 500, Nasdaq-100 and Dow through SPY, QQQ and DIA (index values aren't on the free tier), and a weekly chart with trend figures for any ticker.
- ✅ News (`src/data/news_data.py`): SerpAPI's Google News engine, up to 6 articles from the last 30 days per search, cached 60 minutes, with the last good result kept as a fallback (the free plan allows 250 searches a month). Results have headlines but no summaries, so the agent works from headlines and lists every article as a linked source.
- ⏳ Benchmarks in `docs/benchmarks.md`.

## 7. User accounts (MongoDB)

Sign-up stores one document per user in each of three collections (`src/data/user_store.py`):

```
users                     profiles                        portfolios
─────────────────         ──────────────────────────      ─────────────────────
_id                ◄───── user_id (unique)                user_id (unique) ───► users._id
email (unique)            knowledge_level                 market  "US"
name                      risk_tolerance                  holdings[] (embedded)
password_hash (bcrypt)    interests[]                       ticker, quantity,
created_at                goals[] (embedded)                cost_basis, asset_type
last_login_at               name, target_amount, ...      updated_at
                          updated_at
```

- *Cash* is entered in its own field and stored as a holding (`CASH`, type `cash`, one unit per dollar at $1), the same format as the sample portfolio.
- *Login details are separate* from profile data, so loading a profile never reads the password hash.
- *Goals and holdings are embedded:* small, always read together and saved in one write. They reuse the Pydantic models (`FinancialGoal`, `Holding`), so stored data is validated the same way as anything else.
- *Passwords:* bcrypt hashes only; a failed login takes the same time whether or not the email exists.
- *No transactions:* a standalone MongoDB has none, so if saving the profile or portfolio fails, the new user is deleted again.
- *Portfolio tab:* a signed-in user's holdings are valued at Alpha Vantage prices (cash at $1; a holding without a quote falls back to the price paid, or is listed as unpriced). Gain/loss covers only holdings with a price paid. Users without a portfolio get a link to add one on My account.
- *Signed-in only:* the assistant page (chat, Portfolio and Market tabs) sends anyone logged out to the log-in page; home, sign-up and log-in are public.
- *Session:* logging in puts the account, profile and portfolio in `st.session_state`; the profile personalizes answers, and the portfolio is passed to the workflow. Sidebar changes to knowledge level and risk are saved back.
- Example: signing up as an advanced, aggressive investor with the sample portfolio creates `users {email: "ada@example.com"}`, `profiles {knowledge_level: "advanced", risk_tolerance: "aggressive"}` and `portfolios {market: "US", holdings: [VTI, VXUS, BND, AAPL, MSFT, CASH]}`.
- *Conversations:* LangGraph's `MongoDBSaver` (`default_checkpointer` in `src/workflow/graph.py`) stores each session's graph state in `conversation_checkpoints` and `conversation_writes`, in the same database: message history, a paused question waiting for an answer, and portfolio amounts given for goals. Each checkpoint expires `CONVERSATION_TTL_DAYS` (default 30) after it was written, through a TTL index on `created_at`. They survive restarts and are shared by every app instance, which several ECS tasks need. If MongoDB can't be reached at startup, conversations fall back to process memory with a warning; `CONVERSATION_STORE=memory` chooses memory deliberately.
- *Changing the expiry* of an existing database needs the old TTL index dropped first (`db.conversation_checkpoints.dropIndex("created_at_1")`, same for `conversation_writes`); otherwise MongoDB rejects the new index and conversations fall back to memory.
- The on-screen transcript is still kept in `st.session_state`, so it isn't reloaded after logging in again, even though the agents still have the history.
- ✅ MongoDB on AWS: a free Atlas M0 cluster (created once by hand, outside CDK), with the connection string in Secrets Manager. The instance has an Elastic IP, so Atlas allows a single fixed address.

## 8. Deployment (AWS)

The same Docker image runs locally (`docker compose`) and on AWS. CDK (`infra/`) deploys it to one EC2 instance in us-east-1; the lab account doesn't support ECS or load balancers and only allows `t2.micro`.

```
Allowed IP ─► HTTP :80 ─► EC2 t2.micro ─► Docker: Finnie (Streamlit :8501)
                                            ├─ image from ECR
                                            ├─ API keys from Secrets Manager     
                                            └─ logs to CloudWatch
```

| Resource | Purpose | Lifecycle |
|---|---|---|
| EC2 instance `t2.micro` (Amazon Linux 2023) + 16 GB encrypted gp3 volume | Runs the container; Docker, 1 GB swap and the app start from a boot script | `FinnieStack` |
| Security group | Inbound HTTP :80 only from `allowedCidrs`; no SSH | `FinnieStack` |
| IAM role, policy and instance profile | Pull the image, read the API key secrets, write logs (+ Systems Manager for troubleshooting) | `FinnieStack` |
| Launch template | Enforces IMDSv2 | `FinnieStack` |
| CloudWatch log group | Container logs, 1-month retention | `FinnieStack` |
| ECR repository + S3 bucket | Where CDK pushes the image and uploads templates | `CDKToolkit` (one-time `cdk bootstrap`) |
| Secrets Manager secrets `finnie/google-api-key` (required), `finnie/alpha-vantage-api-key` and `finnie/serp-api-key` (optional) | API keys, fetched at boot into the container's env file; never in code, image or CloudFormation | Created once with the AWS CLI |
| Default VPC and public subnets | Network for the instance | Existing; not created |

- *Deploy:* `cdk deploy -c allowedCidrs=<ip>/32` builds the x86 image, pushes it to ECR and creates the stack; the app is ready ~5 minutes later at the `AppUrl` output.
- *Update:* a new image replaces the instance (a few minutes of downtime, new URL; conversations are lost while they're kept in memory there, see section 7).
- *Credentials:* the lab account doesn't allow assuming roles, so CDK deploys with CLI credentials (`CliCredentialsStackSynthesizer`).
- ⏳ CI/CD (GitHub Actions + OIDC) and HTTPS (needs a load balancer or a domain).

## 9. Open items

- Router test set: 30+ labelled queries, ≥ 90% accuracy.
- Confirm return (4/6/8%, ±2 points) and inflation (2.5%) assumptions; document portfolio score formulas.
