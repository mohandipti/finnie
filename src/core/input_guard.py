"""Input guardrails: checks on the user's message before any agent or LLM sees it.

Deterministic checks live here: personal data is redacted, and over-long messages and prompt
injection attempts are blocked. Off-topic and harmful requests need judgement, so the router
classifies those (RouterDecision.scope). Either way the refusal node answers with one of
REFUSALS instead of running any agent.

The patterns are deliberately narrow: a normal finance question must never be refused, so each
one needs a distinctive phrase ("ignore your previous instructions"), not just a word ("ignore").
"""

import re
from typing import Literal

MAX_QUERY_CHARS = 2000

BlockReason = Literal["too_long", "injection", "off_topic", "disallowed"]

REFUSALS: dict[str, str] = {
    "too_long": (
        f"That message is too long for me to work with. Please shorten it to under {MAX_QUERY_CHARS:,} "
        "characters, or split it into separate questions."
    ),
    "injection": (
        "I can't change how I work or share my instructions. I'm happy to help with personal finance and "
        "investing, though: budgeting, saving, investing, taxes, retirement or your own portfolio."
    ),
    "off_topic": (
        "I can only help with personal finance and investing, such as budgeting, saving, investing, taxes "
        "and retirement. For example, try *\"What is an ETF?\"* or *\"Am I on track for my goal?\"*"
    ),
    "disallowed": (
        "I can't help with that, because it would mean breaking the law or harming someone. I can explain "
        "how the rules work, though, for example what counts as insider trading or tax evasion and why "
        "they're illegal."
    ),
}

PRIVACY_NOTE = (
    "Note: I removed {items} from your message. Please don't share account numbers or identity numbers "
    "here; I never need them to answer."
)

# Attempts to override the system prompt or extract it.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        (r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|these|the)\b"
        r".{0,20}\b(instructions?|prompts?|rules|guidelines|directions)\b"),
        (r"\b(reveal|show|print|output|repeat|display|tell me)\b.{0,30}\b(system|hidden|initial|developer|original)\s+"
        r"(prompt|instructions?|message)\b"),
        r"\bwhat (is|are) your (system prompt|instructions|rules)\b",
        (r"\b(you are now|from now on you are|act as|pretend (to be|you are))\b.{0,40}\b(dan|jailbroken|unrestricted|"
        r"unfiltered|without (any )?(rules|restrictions|limits|filters))\b"),
        r"\b(developer|god|jailbreak|dan) mode\b",
        r"\bjailbreak\b",
    )
]

# Personal data that should never reach the LLM or the conversation store.
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ]\d{2}[- ]\d{4}\b")
_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_ACCOUNT_RE = re.compile(r"\b((?:bank |checking |savings )?(?:account|acct|routing)\s*(?:number|no\.?|#)?\s*:?\s*)(\d{6,17})\b",
                         re.IGNORECASE)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def redact_personal_data(text: str) -> tuple[str, list[str]]:
    """The text with card, Social Security and account numbers replaced, and what was removed."""
    removed: list[str] = []

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if not _luhn_valid(digits):
            return match.group(0)  # a long number that isn't a card, e.g. an amount without commas
        removed.append("a card number")
        return "[card number removed]"

    def ssn(_: re.Match) -> str:
        removed.append("a Social Security number")
        return "[SSN removed]"

    def account(match: re.Match) -> str:
        removed.append("an account number")
        return f"{match.group(1)}[number removed]"

    text = _CARD_RE.sub(card, text)
    text = _SSN_RE.sub(ssn, text)
    text = _ACCOUNT_RE.sub(account, text)
    return text, list(dict.fromkeys(removed))


def privacy_note(removed: list[str]) -> str:
    items = removed[0] if len(removed) == 1 else ", ".join(removed[:-1]) + " and " + removed[-1]
    return PRIVACY_NOTE.format(items=items)


def check_input(text: str) -> BlockReason | None:
    """Why the message should be refused without running any agent, or None if it can go ahead."""
    if len(text) > MAX_QUERY_CHARS:
        return "too_long"
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        return "injection"
    return None
