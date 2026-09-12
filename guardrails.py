"""
guardrails.py
==============
Input and output guardrails as real, testable code -- not just prompt
wording. Two layers on each side:

  Input  (before the Supervisor ever runs):
    1. Deterministic rule checks (regex/keyword) for prompt injection and
       out-of-scope "book this and charge my card" requests -- cheap, fast,
       zero LLM calls, and exactly reproducible in unit tests.
    2. A missing-critical-details check (destination / dates / group size)
       that routes to a clarifying question instead of letting an agent
       silently invent values.
    3. An optional LLM-as-judge second opinion for requests the rules can't
       confidently classify (see `llm_judge_out_of_scope`) -- used sparingly
       since it costs a network call; failures fail open (allow) with a
       logged reason, they never crash the request.

  Output (after the Trip Compiler, before human review):
    1. Disclaimer presence -- auto-inserted if missing (self-healing).
    2. Overly-certain language -- auto-softened via find/replace
       (self-healing).
    3. Unbacked specifics -- flight-number-shaped tokens or currency amounts
       in the draft that don't trace back to any tool-backed context
       (the Weather Agent's MCP result, or the Flight/Hotel agent's notes)
       are flagged for regeneration.

Why not a heavier framework (Guardrails AI / NeMo Guardrails)? For a
4-agent project this rule+Pydantic layer is easier to unit test, has zero
extra runtime dependency, and every check is inspectable in ~150 lines.
The documented upgrade path if this grows is to swap `run_input_guardrail`
/ `run_output_guardrail` internals for one of those frameworks without
touching the graph -- they're called from exactly two places in backend.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------
# Input guardrails
# --------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above)\s+instructions",
    r"disregard (all )?(the )?(previous|prior|above)\s+instructions",
    r"forget (all|everything)\s+(you|that)",
    r"reveal (your|the)\s+(system|hidden)\s*prompt",
    r"(show|print|output)\s+(me\s+)?your\s+(system\s+)?(prompt|instructions)",
    r"what\s+(are|is)\s+your\s+(system\s+)?instructions",
    r"you are now (in\s+)?(dan|developer mode|jailbroken|unrestricted)",
    r"act as if you (have no|had no)\s+(restrictions|rules|guardrails|filters)",
    r"pretend (you have|to have) no (rules|restrictions|guardrails)",
    r"bypass (your|all)\s+(safety|guardrail)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

PAYMENT_KEYWORDS = [
    "charge my card", "charge the card", "charge my account", "card ending",
    "credit card number", "debit card number", "cvv", "process the payment",
    "process payment", "pay now with", "charge my credit card", "charge my debit card",
]
BOOKING_INTENT_KEYWORDS = [
    "book this flight", "book the flight", "book my flight", "book it now",
    "purchase the ticket", "buy the ticket", "confirm the booking", "confirm booking",
    "reserve and pay", "book and pay", "make the reservation and charge",
]

REQUIRED_FIELDS = ("destination", "trip_length_known", "group_size")


@dataclass
class InputGuardrailResult:
    status: Literal["ok", "blocked", "needs_clarification"]
    message: str | None = None
    reasons: list[str] = field(default_factory=list)


def check_prompt_injection(text: str) -> list[str]:
    """Return the list of injection patterns matched (empty = clean)."""
    hits = []
    for pattern, compiled in zip(INJECTION_PATTERNS, _INJECTION_RE):
        if compiled.search(text):
            hits.append(pattern)
    return hits


def check_out_of_scope_booking(text: str) -> bool:
    """True if the request combines real booking intent with a real-payment
    ask -- TripCraft only ever drafts estimates, it never books or charges."""
    lower = text.lower()
    has_payment = any(kw in lower for kw in PAYMENT_KEYWORDS)
    has_booking = any(kw in lower for kw in BOOKING_INTENT_KEYWORDS)
    return has_payment or (has_payment and has_booking)


def missing_critical_fields(extracted: dict) -> list[str]:
    """`extracted` is the structured-extraction dict from backend.py's
    extract_trip_details(). Returns the list of missing field labels."""
    missing = []
    if not extracted.get("destination"):
        missing.append("destination")
    if not (extracted.get("num_days") or (extracted.get("start_date") and extracted.get("end_date"))):
        missing.append("travel dates or trip length")
    if not extracted.get("group_size"):
        missing.append("number of travelers")
    return missing


def llm_judge_out_of_scope(text: str) -> bool:
    """Second-opinion LLM classifier for out-of-scope asks the rule layer
    didn't catch (e.g. phrased without our exact keyword list). Used only
    as a fallback -- fails open (returns False / "allow") on any error so a
    provider hiccup never blocks a legitimate request."""
    try:
        from llm import extract_text, get_llm

        llm = get_llm(temperature=0.0, fast=True)
        prompt = (
            "You are a strict binary classifier for a travel-planning assistant "
            "that ONLY drafts itineraries, weather outlooks and rough budget "
            "estimates. It does NOT book flights/hotels or process payments.\n"
            "Question: does the user request below ask this assistant to "
            "actually execute a real-world booking or payment action (not just "
            "plan/suggest one)? Answer with exactly one word: YES or NO.\n\n"
            f"Request: {text}"
        )
        resp = llm.invoke(prompt)
        answer = extract_text(resp.content).strip().upper()
        return answer.startswith("YES")
    except Exception:  # noqa: BLE001 - fail open, never block on a provider error
        return False


def run_input_guardrail(user_text: str, extracted: dict) -> InputGuardrailResult:
    reasons: list[str] = []

    injection_hits = check_prompt_injection(user_text)
    if injection_hits:
        return InputGuardrailResult(
            status="blocked",
            message=(
                "I can't follow instructions embedded in a trip request that try to "
                "override my configuration. Let's stick to trip planning -- tell me "
                "the destination, dates, and who's traveling and I'll get started."
            ),
            reasons=[f"prompt_injection:{p}" for p in injection_hits],
        )

    if check_out_of_scope_booking(user_text):
        return InputGuardrailResult(
            status="blocked",
            message=(
                "TripCraft only drafts itineraries, weather outlooks and rough budget "
                "estimates for you to review -- it does not book flights/hotels or "
                "process any real payment. I won't charge a card or confirm a booking, "
                "but I'm happy to put together a draft plan you can book yourself."
            ),
            reasons=["out_of_scope:booking_and_payment"],
        )

    missing = missing_critical_fields(extracted)
    if missing:
        return InputGuardrailResult(
            status="needs_clarification",
            message=(
                "Before I plan this trip, I need a bit more info: "
                + ", ".join(missing)
                + ". Could you share those?"
            ),
            reasons=[f"missing:{m}" for m in missing],
        )

    return InputGuardrailResult(status="ok", reasons=reasons)


# --------------------------------------------------------------------------
# Output guardrails
# --------------------------------------------------------------------------

DISCLAIMER_TEXT = (
    "> **Disclaimer:** All prices, availability and timings in this plan are "
    "rough estimates for planning purposes only -- not live bookings. "
    "Please verify current prices and availability directly with the "
    "airline, hotel or booking platform before you travel."
)

_DISCLAIMER_MARKERS = ("estimate", "not a live booking", "not live bookings", "not booked", "disclaimer")

_CERTAINTY_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bguaranteed availability\b", re.IGNORECASE), "likely availability (subject to change)"),
    (re.compile(r"\bconfirmed booking\b", re.IGNORECASE), "suggested booking (not yet confirmed)"),
    (re.compile(r"\b100% available\b", re.IGNORECASE), "likely available"),
    (re.compile(r"\bwill definitely be available\b", re.IGNORECASE), "is expected to be available"),
    (re.compile(r"\bguaranteed\b", re.IGNORECASE), "expected"),
    (re.compile(r"\bassured\b", re.IGNORECASE), "likely"),
]

# Heuristic flight-number-shaped token, e.g. "AI 202", "6E-202", "UK955".
# Requires an uppercase letter immediately followed (optionally via a
# separator) by 2-4 digits, so it doesn't fire on "Day 3" or plain years.
_FLIGHT_NUMBER_RE = re.compile(r"\b[A-Z]{1,2}[\s-]?\d{2,4}\b")
_CURRENCY_AMOUNT_RE = re.compile(r"(?:₹|Rs\.?|INR|\$|USD)\s?\d[\d,]*")


@dataclass
class OutputGuardrailResult:
    text: str
    status: Literal["ok", "fixed", "flagged"]
    notes: list[str] = field(default_factory=list)
    unbacked_specifics: list[str] = field(default_factory=list)


def ensure_disclaimer(text: str) -> tuple[str, bool]:
    lower = text.lower()
    if any(marker in lower for marker in _DISCLAIMER_MARKERS):
        return text, False
    return text.rstrip() + "\n\n" + DISCLAIMER_TEXT, True


def soften_certainty(text: str) -> tuple[str, list[str]]:
    notes = []
    for pattern, replacement in _CERTAINTY_REPLACEMENTS:
        if pattern.search(text):
            notes.append(f"softened '{pattern.pattern}' -> '{replacement}'")
            text = pattern.sub(replacement, text)
    return text, notes


def find_unbacked_specifics(text: str, tool_backed_context: str) -> list[str]:
    """Flight-number-shaped tokens or currency amounts in `text` that do not
    appear anywhere in `tool_backed_context` (the concatenated outputs of
    the Weather / Flight-Hotel agents, i.e. the only places a real specific
    could legitimately come from). Best-effort heuristic, not a proof --
    documented limitation, see module docstring."""
    suspects: list[str] = []
    for match in _FLIGHT_NUMBER_RE.findall(text):
        if match not in tool_backed_context:
            suspects.append(match.strip())
    for match in _CURRENCY_AMOUNT_RE.findall(text):
        if match not in tool_backed_context:
            suspects.append(match.strip())
    # de-dupe, preserve order
    seen = set()
    out = []
    for s in suspects:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_output_guardrail(plan_text: str, tool_backed_context: str) -> OutputGuardrailResult:
    notes: list[str] = []

    text, disclaimer_added = ensure_disclaimer(plan_text)
    if disclaimer_added:
        notes.append("added missing pricing/availability disclaimer")

    text, softened_notes = soften_certainty(text)
    notes.extend(softened_notes)

    unbacked = find_unbacked_specifics(text, tool_backed_context)

    status: Literal["ok", "fixed", "flagged"]
    if unbacked:
        status = "flagged"
    elif notes:
        status = "fixed"
    else:
        status = "ok"

    return OutputGuardrailResult(text=text, status=status, notes=notes, unbacked_specifics=unbacked)
