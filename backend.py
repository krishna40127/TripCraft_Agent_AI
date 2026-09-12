"""
backend.py
===========
The LangGraph orchestration layer: state schema, the Supervisor router, the
four specialized agents, the Trip Compiler, both guardrail nodes, and the
human-in-the-loop interrupt -- wired into one graph and checkpointed with
`MemorySaver` (swap for `SqliteSaver`/`PostgresSaver` for real persistence
across process restarts, see README).

Every LLM call in this file goes through two small seams --
`_call_json()` (structured JSON-in-prompt, parsed back out) and
`_call_text()` (free-form markdown) -- instead of `with_structured_output`
/ tool-calling, because that pattern works identically against Groq-hosted
models and small local Ollama models alike, and it degrades to a documented
fallback instead of crashing if the provider call fails (rate limit, no
network, no API key yet). Tests monkeypatch the module-level
`extract_trip_details` / `run_*_agent` / `supervisor_decide` functions
directly rather than mocking the LLM client, which is more robust across
LangChain/provider versions.
"""
from __future__ import annotations

import calendar
import json
import logging
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from guardrails import run_input_guardrail, run_output_guardrail
from llm import extract_text, get_llm
from mcp_client import get_weather_outlook

logger = logging.getLogger("tripcraft.backend")

MAX_SUPERVISOR_ITERATIONS = 8
MAX_OUTPUT_GUARDRAIL_ATTEMPTS = 2

# --------------------------------------------------------------------------
# State schema
# --------------------------------------------------------------------------


class TripState(TypedDict, total=False):
    user_text: str

    # extracted trip details
    destination: str | None
    origin_city: str | None
    start_date: str | None
    end_date: str | None
    approx_month: str | None
    num_days: int | None
    group_size: int | None
    budget_level: str | None
    interests: list[str]

    # input guardrail
    input_status: Literal["ok", "blocked", "needs_clarification"]
    input_message: str | None

    # resolved dates actually used for the weather lookup
    resolved_start_date: str | None
    resolved_end_date: str | None

    # agent outputs
    weather_report: str | None
    weather_raw: dict | None
    itinerary_draft: str | None
    flight_hotel_notes: str | None
    trip_plan_draft: str | None

    # supervisor bookkeeping
    completed_agents: list[str]
    supervisor_iterations: int
    next_agent: str | None

    # output guardrail
    output_status: Literal["ok", "fixed", "flagged", "flagged_accepted"]
    output_notes: list[str]
    output_guardrail_attempts: int
    unbacked_specifics: list[str]

    # HITL / revision
    revision_feedback: str | None
    status: Literal["pending_clarification", "blocked", "pending_approval", "finalized"]
    final_plan: str | None


# --------------------------------------------------------------------------
# LLM call helpers (JSON-in-prompt / free text, both fail-soft)
# --------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_json(system_prompt: str, user_prompt: str, *, fast: bool = False, default: dict) -> dict:
    try:
        llm = get_llm(temperature=0.1, fast=fast)
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = extract_text(resp.content)
        parsed = _extract_json(content)
        if parsed is None:
            raise ValueError(f"No JSON object found in LLM response: {content[:200]!r}")
        return parsed
    except Exception as e:  # noqa: BLE001 - any provider/parsing failure degrades gracefully
        logger.warning("LLM JSON call failed, using default: %s", e)
        return default


def _call_text(system_prompt: str, user_prompt: str, *, fast: bool = False, default: str) -> str:
    try:
        llm = get_llm(temperature=0.4, fast=fast)
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = extract_text(resp.content).strip()
        return content or default
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM text call failed, using fallback text: %s", e)
        return default


# --------------------------------------------------------------------------
# Trip-detail extraction (runs inside the input guardrail node)
# --------------------------------------------------------------------------

_MONTH_NUM = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_NUM.update({calendar.month_abbr[i].lower(): i for i in range(1, 13)})

EXTRACTION_SYSTEM = """You are a trip-detail extraction assistant for TripCraft, a travel planner.
Read the user's trip request and extract ONLY what is explicitly stated or clearly implied.
Respond with ONLY a single JSON object, no prose, matching exactly this shape:
{
  "destination": string or null,
  "origin_city": string or null,
  "start_date": "YYYY-MM-DD" or null,
  "end_date": "YYYY-MM-DD" or null,
  "approx_month": string or null (e.g. "December", only if an exact start_date wasn't given),
  "num_days": integer or null,
  "group_size": integer or null,
  "budget_level": "budget" or "mid-range" or "luxury" or null,
  "interests": array of short strings (may be empty)
}
Rules:
- Never invent a destination, date, or group size that isn't stated or clearly implied.
- "for 2 people" / "me and my partner" -> group_size 2. "solo trip" -> group_size 1.
  "family of 4" -> group_size 4. If truly unstated, use null.
- "5-day trip" -> num_days 5.
- If only a month/season is given (no exact date), set approx_month and leave start_date/end_date null.
"""


def extract_trip_details(user_text: str) -> dict:
    today_str = date.today().isoformat()
    user_prompt = f"Today's date is {today_str}.\nUser request:\n{user_text}"
    default: dict[str, Any] = {
        "destination": None,
        "origin_city": None,
        "start_date": None,
        "end_date": None,
        "approx_month": None,
        "num_days": None,
        "group_size": None,
        "budget_level": None,
        "interests": [],
    }
    result = _call_json(EXTRACTION_SYSTEM, user_prompt, fast=True, default=default)
    merged = {**default, **result}
    return merged


def _resolve_month_to_year(month_num: int, today: date) -> int:
    year = today.year
    if month_num < today.month or (month_num == today.month and today.day > 25):
        year += 1
    return year


def resolve_date_range(state: TripState) -> tuple[str, str, str | None]:
    """Pick concrete ISO start/end dates for the weather lookup even when the
    user only gave a rough month, or nothing at all -- the MCP weather tool
    needs real dates either way. Returns (start, end, human-readable note)."""
    today = date.today()
    num_days = state.get("num_days") or 5
    start_date = state.get("start_date")
    end_date = state.get("end_date")

    if start_date and end_date:
        return start_date, end_date, None

    if start_date and not end_date:
        s = datetime.strptime(start_date, "%Y-%m-%d").date()
        e = s + timedelta(days=max(num_days - 1, 0))
        return start_date, e.isoformat(), None

    approx_month = (state.get("approx_month") or "").strip().lower()
    month_num = _MONTH_NUM.get(approx_month) or _MONTH_NUM.get(approx_month[:3])
    if month_num:
        year = _resolve_month_to_year(month_num, today)
        s = date(year, month_num, 10)
        e = s + timedelta(days=max(num_days - 1, 0))
        note = f"(No exact dates given -- using {s.isoformat()} as a representative date in {approx_month.title()} {year}.)"
        return s.isoformat(), e.isoformat(), note

    s = today + timedelta(days=30)
    e = s + timedelta(days=max(num_days - 1, 0))
    note = "(No dates or month given -- showing a near-term outlook as a rough proxy.)"
    return s.isoformat(), e.isoformat(), note


# --------------------------------------------------------------------------
# Graph nodes: input guardrail
# --------------------------------------------------------------------------


def input_guardrail_node(state: TripState) -> dict:
    user_text = state.get("user_text", "")
    extracted = extract_trip_details(user_text)
    result = run_input_guardrail(user_text, extracted)

    update: dict[str, Any] = {**extracted, "input_status": result.status, "input_message": result.message}
    if result.status == "blocked":
        update["status"] = "blocked"
    elif result.status == "needs_clarification":
        update["status"] = "pending_clarification"
    return update


def route_after_input_guardrail(state: TripState) -> str:
    return "supervisor" if state.get("input_status") == "ok" else "end_early"


# --------------------------------------------------------------------------
# Graph nodes: Supervisor
# --------------------------------------------------------------------------

SUPERVISOR_AGENTS = ("weather_agent", "itinerary_agent", "flight_hotel_agent", "trip_compiler")

SUPERVISOR_SYSTEM = """You are the Supervisor of TripCraft, a multi-agent trip-planning system.
Decide which single agent should run next. Valid values: weather_agent, itinerary_agent,
flight_hotel_agent, trip_compiler. Pick trip_compiler only once the other agents needed for
this pass have already run (see agents_completed_this_pass). If revision_feedback is present,
prefer re-running only the agents relevant to that feedback -- a rule_based_suggestion is given,
use it unless you have a clearly better reason to override it. Respond with ONLY a JSON object:
{"next": "<one of the valid values>", "reasoning": "<one short sentence>"}"""


def _infer_revision_targets(feedback: str) -> list[str]:
    text = feedback.lower()
    targets: list[str] = []
    if any(k in text for k in ("budget", "price", "cost", "expensive", "cheap", "afford", "money")):
        targets.append("flight_hotel_agent")
    if any(k in text for k in ("day", "itinerary", "activity", "schedule", "relax", "pace", "swap", "plan")):
        targets.append("itinerary_agent")
    if any(k in text for k in ("weather", "rain", "hot", "cold", "season", "timing", "month", "date", "climate")):
        targets.append("weather_agent")
    if not targets:
        targets = ["weather_agent", "itinerary_agent", "flight_hotel_agent"]
    return targets


def supervisor_decide(state: TripState) -> str:
    completed = state.get("completed_agents", [])
    revision_feedback = state.get("revision_feedback")

    if not revision_feedback:
        first_pass_order = ["weather_agent", "itinerary_agent", "flight_hotel_agent"]
        remaining = [a for a in first_pass_order if a not in completed]
        rule_default = remaining[0] if remaining else "trip_compiler"
    else:
        targets = [a for a in _infer_revision_targets(revision_feedback) if a not in completed]
        rule_default = targets[0] if targets else "trip_compiler"

    user_prompt = json.dumps(
        {
            "destination": state.get("destination"),
            "revision_feedback": revision_feedback,
            "agents_completed_this_pass": completed,
            "rule_based_suggestion": rule_default,
        }
    )
    decision = _call_json(SUPERVISOR_SYSTEM, user_prompt, fast=True, default={"next": rule_default})
    next_agent = decision.get("next")
    if next_agent not in SUPERVISOR_AGENTS:
        next_agent = rule_default
    return next_agent


def supervisor_node(state: TripState) -> dict:
    iterations = state.get("supervisor_iterations", 0) + 1
    if iterations > MAX_SUPERVISOR_ITERATIONS:
        return {"next_agent": "trip_compiler", "supervisor_iterations": iterations}
    return {"next_agent": supervisor_decide(state), "supervisor_iterations": iterations}


def route_after_supervisor(state: TripState) -> str:
    return state.get("next_agent") or "trip_compiler"


# --------------------------------------------------------------------------
# Graph nodes: specialized agents
# --------------------------------------------------------------------------


def run_weather_agent(state: TripState) -> dict:
    destination = state.get("destination") or "the destination"
    start_date, end_date, note = resolve_date_range(state)
    result = get_weather_outlook(destination, start_date, end_date)

    if result.get("ok"):
        text = result["text"]
        if note:
            text = f"{text} {note}"
    else:
        text = (
            f"Weather data is temporarily unavailable for {destination} "
            f"({result.get('error', 'unknown error')}). Proceeding without a weather-based "
            "recommendation -- please double-check seasonal conditions yourself before booking."
        )

    return {
        "weather_report": text,
        "weather_raw": result,
        "resolved_start_date": start_date,
        "resolved_end_date": end_date,
        "completed_agents": state.get("completed_agents", []) + ["weather_agent"],
    }


ITINERARY_SYSTEM = """You are TripCraft's Itinerary Agent. Draft a clear, well-paced,
day-by-day itinerary in Markdown for the trip described. Match the stated trip length,
group size, budget level and interests. Include must-see spots and local food
recommendations where relevant. Keep pacing realistic (don't over-pack days). Do not
mention flight numbers, exact hotel names, or exact prices -- another agent handles those.
If revision feedback is given, make the requested change and keep the rest consistent."""


def run_itinerary_agent(state: TripState) -> dict:
    user_prompt = json.dumps(
        {
            "destination": state.get("destination"),
            "origin_city": state.get("origin_city"),
            "num_days": state.get("num_days"),
            "group_size": state.get("group_size"),
            "budget_level": state.get("budget_level"),
            "interests": state.get("interests", []),
            "weather_outlook": state.get("weather_report"),
            "previous_itinerary_draft": state.get("itinerary_draft"),
            "revision_feedback": state.get("revision_feedback"),
        }
    )
    default = (
        f"## Day-by-day itinerary for {state.get('destination') or 'your destination'}\n\n"
        "_Itinerary drafting is temporarily unavailable -- please try again shortly._"
    )
    text = _call_text(ITINERARY_SYSTEM, user_prompt, fast=False, default=default)
    return {
        "itinerary_draft": text,
        "completed_agents": state.get("completed_agents", []) + ["itinerary_agent"],
    }


FLIGHT_HOTEL_SYSTEM = """You are TripCraft's Flight/Hotel Agent. Suggest, in Markdown:
(1) a rough flight approach -- likely route and approximate timing/duration, NOT a specific
flight number or airline booking (this is not a live search); (2) accommodation types/areas
to stay in matching the budget level, NOT a specific hotel brand/property name; (3) a rough
total price RANGE per person and for the group, clearly labeled as an estimate. Never state
an exact confirmed price, exact flight number, or a specific hotel name -- describe options
and ranges only. If revision feedback is given, adjust accordingly."""


def run_flight_hotel_agent(state: TripState) -> dict:
    user_prompt = json.dumps(
        {
            "origin_city": state.get("origin_city"),
            "destination": state.get("destination"),
            "num_days": state.get("num_days"),
            "group_size": state.get("group_size"),
            "budget_level": state.get("budget_level"),
            "previous_notes": state.get("flight_hotel_notes"),
            "revision_feedback": state.get("revision_feedback"),
        }
    )
    default = (
        "## Flights & Stay\n\n_Flight/hotel suggestions are temporarily unavailable -- "
        "please try again shortly._"
    )
    text = _call_text(FLIGHT_HOTEL_SYSTEM, user_prompt, fast=False, default=default)
    return {
        "flight_hotel_notes": text,
        "completed_agents": state.get("completed_agents", []) + ["flight_hotel_agent"],
    }


COMPILER_SYSTEM = """You are TripCraft's Trip Compiler. Synthesize the Weather Outlook,
Day-by-Day Itinerary and Flights & Stay notes given to you into ONE coherent, well-formatted
Markdown trip plan with these sections: Overview, Weather & Timing, Day-by-Day Itinerary,
Flights & Stay, Rough Budget Total, Assumptions Made. Rules:
- Only state specifics (flight numbers, exact hotel names, exact prices) if they literally
  appear in the material given to you -- otherwise describe options/ranges, never invent one.
- Use hedged language for anything uncertain ("likely", "typically", "around") -- never
  "guaranteed" or "confirmed".
- End with a short disclaimer that prices/availability are estimates, not live bookings.
- If revision_feedback is present, make that specific change and briefly note what changed.
- If avoid_tokens is non-empty, do not reuse those exact tokens verbatim -- they could not be
  verified against a real tool result."""


def run_trip_compiler(state: TripState) -> dict:
    user_prompt = json.dumps(
        {
            "destination": state.get("destination"),
            "origin_city": state.get("origin_city"),
            "num_days": state.get("num_days"),
            "group_size": state.get("group_size"),
            "budget_level": state.get("budget_level"),
            "interests": state.get("interests", []),
            "weather_outlook": state.get("weather_report"),
            "itinerary_draft": state.get("itinerary_draft"),
            "flight_hotel_notes": state.get("flight_hotel_notes"),
            "revision_feedback": state.get("revision_feedback"),
            "avoid_tokens": state.get("unbacked_specifics", []),
        }
    )
    default = (
        f"# Trip Plan: {state.get('destination') or 'Your Trip'}\n\n"
        f"{state.get('itinerary_draft') or ''}\n\n{state.get('flight_hotel_notes') or ''}\n\n"
        "_Trip compilation is temporarily unavailable -- showing raw agent notes above._"
    )
    text = _call_text(COMPILER_SYSTEM, user_prompt, fast=False, default=default)
    return {
        "trip_plan_draft": text,
        "completed_agents": state.get("completed_agents", []) + ["trip_compiler"],
    }


# --------------------------------------------------------------------------
# Graph nodes: output guardrail
# --------------------------------------------------------------------------


def output_guardrail_node(state: TripState) -> dict:
    plan = state.get("trip_plan_draft", "") or ""
    tool_context = "\n".join(filter(None, [state.get("weather_report"), state.get("flight_hotel_notes")]))
    result = run_output_guardrail(plan, tool_context)
    attempts = state.get("output_guardrail_attempts", 0) + 1

    if result.status == "flagged" and attempts <= MAX_OUTPUT_GUARDRAIL_ATTEMPTS:
        return {
            "trip_plan_draft": result.text,
            "output_status": "flagged",
            "output_notes": result.notes,
            "output_guardrail_attempts": attempts,
            "unbacked_specifics": result.unbacked_specifics,
            # force the compiler to run again, not the full supervisor loop
            "completed_agents": [a for a in state.get("completed_agents", []) if a != "trip_compiler"],
        }

    final_status = "flagged_accepted" if result.status == "flagged" else result.status
    return {
        "trip_plan_draft": result.text,
        "output_status": final_status,
        "output_notes": result.notes,
        "output_guardrail_attempts": attempts,
        "unbacked_specifics": [],
    }


def route_after_output_guardrail(state: TripState) -> str:
    return "trip_compiler" if state.get("output_status") == "flagged" else "human_review"


# --------------------------------------------------------------------------
# Graph nodes: human-in-the-loop
# --------------------------------------------------------------------------


def human_review_node(state: TripState) -> dict:
    decision = interrupt(
        {
            "draft_plan": state.get("trip_plan_draft"),
            "status": "pending_approval",
        }
    )

    if isinstance(decision, dict) and decision.get("decision") == "approve":
        return {"status": "finalized", "final_plan": state.get("trip_plan_draft")}

    feedback = decision.get("feedback", "") if isinstance(decision, dict) else ""
    return {
        "status": "pending_approval",
        "revision_feedback": feedback,
        "completed_agents": [],
        "output_guardrail_attempts": 0,
        "supervisor_iterations": 0,
    }


def route_after_human_review(state: TripState) -> str:
    return END if state.get("status") == "finalized" else "supervisor"


def end_early_node(state: TripState) -> dict:
    # No-op node: input_status/status are already set by input_guardrail_node.
    return {}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("end_early", end_early_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather_agent", run_weather_agent)
    graph.add_node("itinerary_agent", run_itinerary_agent)
    graph.add_node("flight_hotel_agent", run_flight_hotel_agent)
    graph.add_node("trip_compiler", run_trip_compiler)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", route_after_input_guardrail, {"supervisor": "supervisor", "end_early": "end_early"}
    )
    graph.add_edge("end_early", END)

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "weather_agent": "weather_agent",
            "itinerary_agent": "itinerary_agent",
            "flight_hotel_agent": "flight_hotel_agent",
            "trip_compiler": "trip_compiler",
        },
    )
    graph.add_edge("weather_agent", "supervisor")
    graph.add_edge("itinerary_agent", "supervisor")
    graph.add_edge("flight_hotel_agent", "supervisor")
    graph.add_edge("trip_compiler", "output_guardrail")

    graph.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {"trip_compiler": "trip_compiler", "human_review": "human_review"},
    )
    graph.add_conditional_edges("human_review", route_after_human_review, {END: END, "supervisor": "supervisor"})

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Module-level singleton graph, shared across requests (checkpointer holds
# per-thread state in memory -- swap for SqliteSaver/PostgresSaver for a
# real deployment, see README "State persistence").
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def new_thread_id() -> str:
    return str(uuid.uuid4())


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt_payload(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def start_trip(user_text: str, thread_id: str | None = None) -> dict:
    """Run the graph from scratch for a new trip request until it either
    hits the human-review interrupt, or ends early (blocked / needs
    clarification). Returns a plain dict describing what happened."""
    thread_id = thread_id or new_thread_id()
    graph = get_graph()
    initial_state: TripState = {"user_text": user_text, "completed_agents": [], "supervisor_iterations": 0}
    result = graph.invoke(initial_state, config=_config(thread_id))

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {
            "thread_id": thread_id,
            "status": "pending_approval",
            "draft_plan": interrupt_payload.get("draft_plan"),
        }

    if result.get("status") == "blocked":
        return {"thread_id": thread_id, "status": "blocked", "message": result.get("input_message")}
    if result.get("status") == "pending_clarification":
        return {
            "thread_id": thread_id,
            "status": "pending_clarification",
            "message": result.get("input_message"),
        }
    # Should not normally happen (graph always interrupts before finishing on
    # a fresh run), but degrade gracefully instead of raising.
    return {"thread_id": thread_id, "status": result.get("status", "unknown"), "draft_plan": result.get("trip_plan_draft")}


def submit_decision(thread_id: str, decision: str, feedback: str | None = None) -> dict:
    """Resume a paused thread with the human's approve/revise decision."""
    graph = get_graph()
    resume_payload = {"decision": decision, "feedback": feedback or ""}
    result = graph.invoke(Command(resume=resume_payload), config=_config(thread_id))

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {
            "thread_id": thread_id,
            "status": "pending_approval",
            "draft_plan": interrupt_payload.get("draft_plan"),
        }

    if result.get("status") == "finalized":
        return {"thread_id": thread_id, "status": "finalized", "final_plan": result.get("final_plan")}

    return {"thread_id": thread_id, "status": result.get("status", "unknown"), "draft_plan": result.get("trip_plan_draft")}
