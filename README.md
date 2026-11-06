# Finnie — AI Finance Assistant

A multi-agent assistant that teaches personal finance and investing basics. Design: [docs/DESIGN.md](docs/DESIGN.md) · Specification: [SPEC.md](SPEC.md).

> Finnie provides educational information only and is not financial, investment, tax or legal advice.

## Setup

**Requirements:** Python 3.11 or newer, and a Google AI Studio API key ([get one](https://aistudio.google.com/apikey)).

**macOS / Linux**

```bash
./scripts/setup.sh           # creates .venv, installs dependencies, creates .env
source .venv/bin/activate
```

**Windows (PowerShell) or manual setup**

```bash
python -m venv .venv
.venv\Scripts\activate       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # macOS/Linux: cp .env.example .env
```

Then open `.env` and set `GOOGLE_API_KEY`. `ALPHA_VANTAGE_API_KEY` is optional; without it there are no live prices: the Portfolio tab values holdings at the price paid, and the Market tab and Market agent say the data is unavailable. `SERP_API_KEY` is optional too; without it the News agent says news is unavailable.

**Lighter install:** `pip install -e .` installs only the runtime dependencies. Add `[dev]` for tests and linting, or `[local-embeddings]` for offline embeddings (`EMBEDDING_PROVIDER=local`, downloads PyTorch).

## Run

```bash
finnie-index --rebuild   # build the knowledge base index (or: python -m src.rag.index --rebuild)
finnie        # launches the Streamlit app
pytest        # runs the tests with coverage
```

## User accounts (MongoDB)

Sign-up and log-in store accounts in MongoDB: the login details, the profile (knowledge level, risk tolerance, interests, goals) and the portfolio. Finnie itself is for signed-in users only: the home, sign-up and log-in pages are public, and opening the assistant while logged out goes to the log-in page.

I point `MONGODB_URI` in `.env` at any MongoDB (default `mongodb://localhost:27017`, database `finnie`):

- **Already installed locally** (for example with Homebrew): nothing else to do.
- **Not installed:** `docker compose up -d mongo` starts one in Docker (`MONGO_PORT=27018` if 27017 is taken).

If MongoDB can't be reached, the sign-up and log-in pages say so and the rest of the app keeps working.

Conversations are stored in the same database (`conversation_checkpoints` and `conversation_writes`), so a restart doesn't lose them, and they're deleted 30 days after they were written (`CONVERSATION_TTL_DAYS`). They include what users tell Finnie about their goals and portfolio. If MongoDB is down when Finnie starts, conversations are kept in memory instead until the next restart; `CONVERSATION_STORE=memory` does that on purpose.

## Run with Docker

The same image is used locally and for deployment. Build the knowledge base index first (it is copied into the image), then:

```bash
docker compose up --build        # open http://localhost:8501
FINNIE_PORT=8502 docker compose up --build   # if 8501 is already in use
docker compose down              # stop and remove the container
```

Settings and secrets are read from `.env` at run time; `.env` is never copied into the image. After adding or editing knowledge base articles, run `finnie-index --rebuild` and rebuild the image.

## Deploy to AWS

See [infra/README.md](infra/README.md) for deploying the Docker image to AWS (EC2) with CDK.
