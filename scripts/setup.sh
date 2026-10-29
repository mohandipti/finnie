#!/usr/bin/env bash
# Create a local virtual environment and install Finnie with all development extras.
# Usage: ./scripts/setup.sh            (uses python3.12, then python3.11, then python3)
#        PYTHON=python3.11 ./scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
VENV_DIR=".venv"

if [[ -z "${PYTHON:-}" ]]; then
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
    done
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "Python 3.11+ not found. Install it, or set PYTHON=/path/to/python." >&2
    exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "$PYTHON is $("$PYTHON" --version 2>&1); Finnie needs Python 3.11 or newer." >&2
    exit 1
fi

echo "Creating $VENV_DIR with $("$PYTHON" --version)..."
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[dev,local-embeddings]"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Add your GOOGLE_API_KEY to it."
fi

cat <<EOF

Setup complete. Next steps:
  source $VENV_DIR/bin/activate
  # edit .env and set GOOGLE_API_KEY
  pytest            # run the tests
  finnie            # launch the app
EOF
