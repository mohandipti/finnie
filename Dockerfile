# Finnie: Streamlit app + LangGraph workflow + FAISS knowledge base.
#
#   docker build -t finnie .
#   docker run -p 8501:8501 --env-file .env finnie
#
# The FAISS index (data/index/) is copied into the image, so build it first with
# `finnie-index --rebuild`. Secrets such as GOOGLE_API_KEY are passed at run time
# and are never baked into the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first, in their own layer, so code changes don't reinstall them.
# Runtime dependencies only: no dev tools, and no local-embeddings extra (PyTorch).
COPY pyproject.toml README.md ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" \
        > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Application code and data. Installed in editable mode so paths resolve to /app,
# where the knowledge base, sample data and index live.
COPY src ./src
COPY mcp_server ./mcp_server
COPY data/index ./data/index
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 1000 finnie \
    && chown -R finnie:finnie /app
USER finnie

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "src/web_app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
