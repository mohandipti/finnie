"""Standard disclaimer and advice-like phrasing checked by the compliance guard."""

import re

DISCLAIMER = (
    "Finnie provides educational information only and is not financial, investment, tax or legal advice. "
    "Consult a licensed professional before making financial decisions."
)

ADVICE_CAVEAT = (
    "Note: this is general education, not a recommendation to buy, sell or hold any investment."
)

# Phrasing that reads like personal advice or a promise about returns.
ADVICE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byou should (definitely )?(buy|sell|hold|invest in|short)\b",
        r"\b(i|we) (recommend|suggest) (that you )?(buy|sell|hold|invest)",
        r"\bguaranteed (return|profit|gain)s?\b",
        r"\b(will|is going to) (definitely|certainly) (rise|go up|increase|fall|go down)\b",
        r"\b(risk[- ]free|can'?t lose)\b",
    )
]


_NEGATION = re.compile(r"\b(not|never|no|isn't|aren't|nothing is)\b[\w\s,]{0,20}$", re.IGNORECASE)


def find_advice_phrases(text: str) -> list[str]:
    """Advice-like phrases, ignoring negated ones such as "ETFs are not risk-free"."""
    return [
        m.group(0)
        for pattern in ADVICE_PATTERNS
        for m in pattern.finditer(text)
        if not _NEGATION.search(text[max(0, m.start() - 30) : m.start()])
    ]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def has_disclaimer(text: str) -> bool:
    """True if the standard disclaimer is present, ignoring punctuation and formatting differences."""
    return _normalize(DISCLAIMER) in _normalize(text)


def strip_disclaimer(text: str) -> str:
    """Remove the appended disclaimer footer, e.g. before storing an answer in conversation history."""
    marker = f"\n\n---\n*{DISCLAIMER}*"
    return text.split(marker)[0].rstrip()
