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

Then open `.env` and set `GOOGLE_API_KEY`. `ALPHA_VANTAGE_API_KEY` is optional; without it, sample market data is used.

**Lighter install:** `pip install -e .` installs only the runtime dependencies. Add `[dev]` for tests and linting, or `[local-embeddings]` for offline embeddings (`EMBEDDING_PROVIDER=local`, downloads PyTorch).

## Run

```bash
finnie-index --rebuild   # build the knowledge base index (or: python -m src.rag.index --rebuild)
finnie        # launches the Streamlit app
pytest        # runs the tests with coverage
```

## Run with Docker

The same image is used locally and for deployment. Build the knowledge base index first (it is copied into the image), then:

```bash
docker compose up --build        # open http://localhost:8501
FINNIE_PORT=8502 docker compose up --build   # if 8501 is already in use
docker compose down              # stop and remove the container
```

Settings and secrets are read from `.env` at run time; `.env` is never copied into the image. After adding or editing knowledge base articles, run `finnie-index --rebuild` and rebuild the image.
