"""Load sample user data from static files.

Temporary: these files stand in for real user data until sign-up is added.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.core.models import UserProfile

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
SAMPLE_PROFILES_DIR = DATA_DIR / "sample_profiles"
SAMPLE_PORTFOLIO_FILE = DATA_DIR / "sample_portfolios" / "sample_portfolio.json"
SAMPLE_MARKET_FILE = DATA_DIR / "sample_market" / "overview.json"


def load_sample_portfolio() -> dict[str, Any]:
    """Sample holdings with fixed prices and expense ratios, for the Portfolio tab."""
    return json.loads(SAMPLE_PORTFOLIO_FILE.read_text())


def load_sample_market() -> dict[str, Any]:
    """Sample index levels and price-path parameters, for the Market tab."""
    return json.loads(SAMPLE_MARKET_FILE.read_text())


def load_sample_profile(session_id: str, name: str = "default") -> UserProfile:
    """Load `sample_profiles/<name>.json` for this session.

    Falls back to a plain beginner profile if the file is missing or invalid, so a bad
    sample file never stops the conversation.
    """
    path = SAMPLE_PROFILES_DIR / f"{name}.json"
    try:
        profile = UserProfile.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("could not load sample profile %s, using defaults: %s", path, exc)
        return UserProfile(session_id=session_id)
    return profile.model_copy(update={"session_id": session_id})
