"""
mcp_client.py
==============
Thin helper around the official MCP Python SDK that lets the rest of
TripCraft (the Weather Agent in backend.py) call
`custom_weather_mcp_server.py` the *real* way: spawning it as a subprocess
and talking MCP-over-stdio, exactly like a general-purpose MCP client
(Claude Desktop, etc.) would. This keeps the weather tool genuinely
decoupled behind the protocol instead of being a plain Python import.

Usage:
    from mcp_client import get_weather_outlook

    result = get_weather_outlook("Goa, India", "2025-12-10", "2025-12-14")
    print(result["text"])

Standalone manual test (spec Phase 1 requirement -- exercise the client
against the real server subprocess before wiring into agents):
    python mcp_client.py "Goa, India" 2025-12-10 2025-12-14
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ContentBlock, TextContent

_SERVER_SCRIPT = str(Path(__file__).parent / "custom_weather_mcp_server.py")


def _first_text(blocks: list[ContentBlock], default: str) -> str:
    """MCP tool results are a list of content blocks -- text, image, audio,
    resource link, or embedded resource -- only TextContent has `.text`.
    Our weather tool only ever returns text (a JSON-encoded string), but we
    narrow the type properly instead of assuming block[0] is TextContent."""
    for block in blocks:
        if isinstance(block, TextContent):
            return block.text
    return default


async def _call_weather_tool_async(destination: str, start_date: str, end_date: str) -> dict[str, Any]:
    server_params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_weather_outlook",
                {"destination": destination, "start_date": start_date, "end_date": end_date},
            )
            if result.isError:
                text = _first_text(result.content, "Unknown MCP tool error.")
                return {"ok": False, "error": text}
            # FastMCP returns tool results as a JSON-encoded text block.
            payload = _first_text(result.content, "")
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {"ok": False, "error": f"Could not parse MCP tool response: {payload}"}


def get_weather_outlook(destination: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Synchronous wrapper: spawn the weather MCP server, call its one tool,
    return the parsed result dict. Safe to call from sync agent code.

    Degrades gracefully on any transport/tool failure -- callers should check
    result["ok"] rather than assume success, per the spec's "fail soft, don't
    crash the graph" requirement.
    """
    try:
        return asyncio.run(_call_weather_tool_async(destination, start_date, end_date))
    except Exception as e:  # noqa: BLE001 - MCP transport errors, timeouts, etc.
        return {"ok": False, "error": f"Weather data temporarily unavailable: {e}"}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <destination> <start_date YYYY-MM-DD> <end_date YYYY-MM-DD>")
        sys.exit(1)
    dest, sd, ed = sys.argv[1], sys.argv[2], sys.argv[3]
    out = get_weather_outlook(dest, sd, ed)
    print(json.dumps(out, indent=2))
