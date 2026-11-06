"""ASGI entry point: the Streamlit app plus a session cookie, so a page refresh keeps the user signed in.

Run with `finnie` or `streamlit run src/web_app/server.py`. Running app.py directly still works,
but without the cookie a page refresh signs the user out.
"""

import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allows `streamlit run` without installing the package
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import get_settings
from src.data.session_store import SESSION_COOKIE

APP_SCRIPT = Path(__file__).resolve().parent / "app.py"
# Streamlit's own endpoints and assets; the cookie is set when a page is loaded.
INTERNAL_PREFIXES = ("/_stcore/", "/static/", "/media/", "/component/")


class SessionCookieMiddleware(BaseHTTPMiddleware):
    """Gives each browser a random, HttpOnly session cookie the first time it loads a page.

    The cookie only identifies the browser. Whether it is signed in is decided by the server-side
    session in MongoDB, which is created at log-in and deleted at log-out.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if SESSION_COOKIE not in request.cookies and not request.url.path.startswith(INTERNAL_PREFIXES):
            https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            response.set_cookie(
                SESSION_COOKIE, secrets.token_urlsafe(32), max_age=get_settings().session_ttl_days * 24 * 3600,
                httponly=True, samesite="lax", secure=https,
            )
        return response


app = st.App(APP_SCRIPT, middleware=[Middleware(SessionCookieMiddleware)])
