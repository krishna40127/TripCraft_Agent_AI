"""
app.py
=======
FastAPI web server exposing TripCraft's HITL travel-planning flow:

  POST /api/travel          submit a new trip request
  POST /api/travel/approve  approve or request changes on a pending draft
  GET  /health               liveness check
  GET  /                     chat + approve/revise UI

Route handlers are plain sync `def`s, so Starlette already runs each one in
its own worker thread (via `run_in_threadpool`) rather than on the main
event loop. backend.py's `graph.invoke(...)` (LangGraph's sync API) and
mcp_client.py's internal `asyncio.run(...)` (spawning the weather MCP
subprocess) both need a thread with no event loop already running, which is
exactly what that worker thread is -- no event-loop bridging required.
`nest_asyncio` is intentionally NOT applied here: patching the loop that
way conflicts with anyio's own thread-local loop detection on this stack
and breaks routing. If a future change moves these calls onto `async def`
routes, reach for `anyio.to_thread.run_sync` instead of nest_asyncio.
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripcraft.app")

import backend  # noqa: E402 - after dotenv/nest_asyncio setup

app = FastAPI(title="TripCraft", description="Multi-agent AI travel planner")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TravelRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Free-text trip request")


class TravelResponse(BaseModel):
    thread_id: str
    status: str
    draft_plan: str | None = None
    message: str | None = None


class ApproveRequest(BaseModel):
    thread_id: str
    decision: str = Field(..., pattern="^(approve|revise)$")
    feedback: str | None = None


class ApproveResponse(BaseModel):
    thread_id: str
    status: str
    draft_plan: str | None = None
    final_plan: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": backend_provider_name()}


def backend_provider_name() -> str:
    from llm import provider_name

    return provider_name()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Starlette >=0.28 wants `request` as the first positional arg (the old
    # `TemplateResponse(name, {"request": request})` calling convention is
    # gone and silently produces a broken jinja2 cache-key error instead).
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/api/travel", response_model=TravelResponse)
def submit_travel_request(payload: TravelRequest):
    try:
        result = backend.start_trip(payload.message)
    except Exception as e:  # noqa: BLE001 - never 500 the whole request on an agent hiccup
        logger.exception("start_trip failed")
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {e}") from e
    return TravelResponse(
        thread_id=result["thread_id"],
        status=result["status"],
        draft_plan=result.get("draft_plan"),
        message=result.get("message"),
    )


@app.post("/api/travel/approve", response_model=ApproveResponse)
def approve_travel_request(payload: ApproveRequest):
    try:
        result = backend.submit_decision(payload.thread_id, payload.decision, payload.feedback)
    except Exception as e:  # noqa: BLE001
        logger.exception("submit_decision failed")
        raise HTTPException(status_code=500, detail=f"Could not process decision: {e}") from e
    return ApproveResponse(
        thread_id=result["thread_id"],
        status=result["status"],
        draft_plan=result.get("draft_plan"),
        final_plan=result.get("final_plan"),
    )


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
