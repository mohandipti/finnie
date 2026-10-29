"""`finnie` command: launches the Streamlit app."""

import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "web_app" / "app.py"


def main() -> None:
    if not APP_PATH.exists():
        sys.exit(f"Streamlit app not found at {APP_PATH}. It hasn't been implemented yet.")

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
